"""
server.py — HTTP backend for the Electron turret app.

Routes:
    GET  /                → index.html
    GET  /config          → {cameras, targets, presets}  (cameras include JPEG snapshots)
    POST /launch          → start tracker with chosen config
    POST /stop            → stop tracker
    POST /key             → inject a keypress into the running tracker
    GET  /status          → {running}
    GET  /stream          → MJPEG stream of the live tracker frames

Spawned by Electron's main.js; lives for the lifetime of the app.
"""

import base64
import http.server
import json
import socketserver
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2

import turret_track as tracker


ROOT       = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "index.html"
PORT       = 8765
MAX_PROBE  = 4


TARGETS = [
    {"id": "face",      "label": "Face",         "model": "yolov8n-face.pt", "class": None},
    {"id": "person",    "label": "Person",       "model": "yolov8n.pt",      "class": "person"},
    {"id": "bottle",    "label": "Water bottle", "model": "yolov8n.pt",      "class": "bottle"},
    {"id": "cup",       "label": "Cup",          "model": "yolov8n.pt",      "class": "cup"},
    {"id": "cellphone", "label": "Cell phone",   "model": "yolov8n.pt",      "class": "cell phone"},
    {"id": "book",      "label": "Book",         "model": "yolov8n.pt",      "class": "book"},
]

PRESETS = [
    {"id": 0, "label": "640 x 480  (narrowest FOV)"},
    {"id": 1, "label": "1280 x 720"},
    {"id": 2, "label": "1920 x 1080  (widest FOV)"},
]


def probe_and_snapshot(max_idx=MAX_PROBE):
    """One snapshot per working camera, as base64 JPEG."""
    snaps = []
    for i in range(max_idx + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        for _ in range(4):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            continue
        frame = cv2.resize(frame, (320, 240))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        snaps.append({
            "index":    i,
            "jpeg_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
        })
    return snaps


SNAPSHOTS = []


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a, **kw):
        pass

    # --- GET --------------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._serve_file(INDEX_HTML, "text/html; charset=utf-8")
        if path == "/config":
            return self._serve_json({
                "cameras": SNAPSHOTS,
                "targets": TARGETS,
                "presets": PRESETS,
            })
        if path == "/status":
            return self._serve_json({
                "running": tracker.is_running(),
                "active":  tracker.get_tracking_active(),
            })
        if path == "/stream":
            return self._serve_mjpeg()
        self.send_response(404); self.end_headers()

    # --- POST -------------------------------------------------------------
    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/launch":
            return self._handle_launch()
        if path == "/stop":
            tracker.stop()
            return self._serve_json({"ok": True})
        if path == "/key":
            return self._handle_key()
        if path == "/mode":
            return self._handle_mode()
        self.send_response(404); self.end_headers()

    # --- handlers ---------------------------------------------------------
    def _handle_launch(self):
        body = self._read_json()
        if body is None or "cam_index" not in body:
            self.send_response(400); self.end_headers(); return
        target = next((t for t in TARGETS if t["id"] == body.get("target_id")),
                      TARGETS[0])
        cfg = {
            "cam_index":      int(body["cam_index"]),
            "capture_preset": int(body.get("capture_preset", 0)),
            "model_path":     target["model"],
            "target_class":   target["class"] or "",
        }
        ok, msg = tracker.start(cfg)
        self._serve_json({"ok": ok, "msg": msg})

    def _handle_key(self):
        body = self._read_json()
        if not body or "key" not in body:
            self.send_response(400); self.end_headers(); return
        tracker.inject_key(str(body["key"]))
        self._serve_json({"ok": True})

    def _handle_mode(self):
        body = self._read_json()
        if body is None or "active" not in body:
            self.send_response(400); self.end_headers(); return
        tracker.set_tracking_active(bool(body["active"]))
        self._serve_json({"ok": True, "active": tracker.get_tracking_active()})

    # --- MJPEG stream -----------------------------------------------------
    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=FRAME"
        )
        self.end_headers()
        last_id = None
        try:
            while True:
                jpeg = tracker.get_latest_jpeg()
                jpeg_id = id(jpeg)
                if jpeg is None or jpeg_id == last_id:
                    time.sleep(0.02)
                    if not tracker.is_running() and jpeg is None:
                        # Tracker stopped and no frame to show — push a blank.
                        time.sleep(0.1)
                    continue
                last_id = jpeg_id
                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"[stream] disconnect: {e}")

    # --- utils ------------------------------------------------------------
    def _read_json(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return None

    def _serve_file(self, path, content_type):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads      = True


def main():
    global SNAPSHOTS
    if not INDEX_HTML.exists():
        print(f"Missing {INDEX_HTML.name} next to server.py.")
        return 1

    print("[server] probing cameras...")
    SNAPSHOTS = probe_and_snapshot()
    print(f"[server] found cameras at indices: "
          f"{[s['index'] for s in SNAPSHOTS]}")

    server = _Server(("127.0.0.1", PORT), Handler)
    print(f"[server] listening on http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
        server.server_close()


if __name__ == "__main__":
    sys.exit(main() or 0)
