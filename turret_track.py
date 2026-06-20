"""
turret_track.py — closed-loop face/object tracking turret over USB serial.

Two entry points:
  • Importable API (used by server.py for the Electron UI):
        start(config)        → spawns tracker thread
        stop()               → asks it to stop
        inject_key(c)        → queue a key like 'q'/'z'/'+'/'-'
        get_latest_jpeg()    → bytes of the most recent rendered frame
  • Direct run (standalone, picks camera via env vars or the old picker):
        python turret_track.py

Pipeline: camera → YOLO → P-controller → servo angles on <pan,tilt,0>\\n via USB.
"""

import atexit
import io
import os
import queue
import threading
import time
import traceback

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None

try:
    import serial
except ImportError:
    serial = None


# ─── CONFIG (constants only — runtime values live in module state) ──────────

PORT = "/dev/cu.usbserial-110"
BAUD = 115200

DEADZONE_PX         = 30
KP                  = 0.04
MAX_STEP_DEG        = 10.0
SEND_EPS_DEG        = 0.5
MIN_SEND_INTERVAL_S = 0.04
HEAD_OFFSET_FRAC    = 0.25
AIM_EMA_ALPHA       = 0.5

INVERT_PAN  = False
INVERT_TILT = False

INFER_SIZE      = 320
CAPTURE_PRESETS = [(640, 480), (1280, 720), (1920, 1080)]
FRAME_W, FRAME_H = 640, 480

COLOR_TARGET    = (80, 255, 120)
COLOR_OTHER     = (70, 70, 230)
COLOR_HUD       = (240, 240, 240)
COLOR_ACCENT    = (0, 200, 255)
COLOR_PANEL     = (15, 15, 15)
COLOR_CROSSHAIR = (0, 220, 255)
COLOR_LOCK      = (60, 255, 60)


# ─── MODULE STATE ───────────────────────────────────────────────────────────

cx, cy = FRAME_W // 2, FRAME_H // 2

# Set per-launch from the config dict in start().
pan  = 90.0
tilt = 90.0
capture_idx       = 0
cap_w = cap_h     = 0
conf_threshold    = 0.5
head_offset_frac  = HEAD_OFFSET_FRAC
target_class_idx  = None    # None = no class filter (e.g. face model)
device            = "cpu"   # filled in start()
model             = None    # filled in start()
tracking_active   = False   # PASSIVE by default: detect + draw, but don't drive servos

# Threading + IO buffers
_stop_event   = threading.Event()
_key_queue    = queue.Queue(maxsize=32)
_latest_jpeg  = None
_jpeg_lock    = threading.Lock()
_worker       = None

# Serial — kept open across launches so re-arming is instant.
ser = None
if serial is None:
    print("[warn] pyserial not installed — running vision-only.")
else:
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1, write_timeout=0.05)
        time.sleep(2)
    except Exception as e:
        print(f"[warn] serial open failed ({e}) — running vision-only.")

last_sent   = (None, None)
last_send_t = 0.0


# ─── HELPERS ────────────────────────────────────────────────────────────────

def open_camera(idx):
    global cap_w, cap_h
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return None
    req_w, req_h = CAPTURE_PRESETS[capture_idx]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap


def send_servo(p, t):
    global last_sent, last_send_t
    if ser is None:
        return
    now = time.time()
    if now - last_send_t < MIN_SEND_INTERVAL_S:
        return
    rp, rt = round(p, 1), round(t, 1)
    lp, lt = last_sent
    if lp is not None and abs(rp - lp) < SEND_EPS_DEG and abs(rt - lt) < SEND_EPS_DEG:
        return
    try:
        ser.write(f"<{p:.1f},{t:.1f},0>\n".encode())
        last_sent   = (rp, rt)
        last_send_t = now
    except Exception as e:
        print(f"[warn] serial write failed: {e}")


def _draw_panel(frame, x1, y1, x2, y2, alpha=0.55):
    h, w = frame.shape[:2]
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(w, x2), min(h, y2)
    if rx2 > rx1 and ry2 > ry1:
        roi = frame[ry1:ry2, rx1:rx2]
        fill = np.empty_like(roi)
        fill[:] = COLOR_PANEL
        cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, dst=roi)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ACCENT, 1)


def _draw_crosshair(frame, locked):
    ring = COLOR_LOCK if locked else COLOR_CROSSHAIR
    cv2.circle(frame, (cx, cy), DEADZONE_PX, ring, 1)
    cv2.line(frame, (cx - 25, cy), (cx - 8, cy), COLOR_CROSSHAIR, 1)
    cv2.line(frame, (cx + 8, cy), (cx + 25, cy), COLOR_CROSSHAIR, 1)
    cv2.line(frame, (cx, cy - 25), (cx, cy - 8), COLOR_CROSSHAIR, 1)
    cv2.line(frame, (cx, cy + 8), (cx, cy + 25), COLOR_CROSSHAIR, 1)
    cv2.circle(frame, (cx, cy), 2, COLOR_CROSSHAIR, -1)


def _draw_target(frame, box, conf):
    x1, y1, x2, y2 = box
    L, t = 18, 2
    for a, b in [((x1, y1), (x1 + L, y1)), ((x1, y1), (x1, y1 + L)),
                 ((x2, y1), (x2 - L, y1)), ((x2, y1), (x2, y1 + L)),
                 ((x1, y2), (x1 + L, y2)), ((x1, y2), (x1, y2 - L)),
                 ((x2, y2), (x2 - L, y2)), ((x2, y2), (x2, y2 - L))]:
        cv2.line(frame, a, b, COLOR_TARGET, t)
    label = f"TARGET  {conf * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), COLOR_TARGET, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def _draw_other(frame, box, conf):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OTHER, 1)
    cv2.putText(frame, f"{conf * 100:.0f}%", (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_OTHER, 1)


def _draw_status(frame, fps, cam_idx, link_ok, num_faces,
                  cap_ms, inf_ms, loop_ms):
    _draw_panel(frame, 8, 8, 252, 104)
    cv2.putText(frame, "TURRET", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1)
    mode_txt = "ACTIVE" if tracking_active else "PASSIVE"
    mode_col = COLOR_LOCK if tracking_active else COLOR_OTHER
    cv2.putText(frame, mode_txt, (98, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, mode_col, 1)
    link = "LINK" if link_ok else "NO LINK"
    cv2.putText(frame, f"{link}   {device}   conf {conf_threshold:.2f}",
                (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_HUD, 1)
    cv2.putText(frame,
                f"cam {cam_idx}   {cap_w}x{cap_h}   {fps:4.1f}fps   F{num_faces}",
                (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HUD, 1)
    cv2.putText(frame,
                f"cap {cap_ms:4.0f}   inf {inf_ms:4.0f}   loop {loop_ms:4.0f} ms",
                (16, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_ACCENT, 1)


def _draw_controls(frame):
    _draw_panel(frame, FRAME_W - 172, 8, FRAME_W - 8, 88)
    cv2.putText(frame, "CONTROLS", (FRAME_W - 164, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1)
    rows = [("Q",   "stop"), ("+/-", "conf threshold"), ("Z",   "zoom out")]
    for i, (k, v) in enumerate(rows):
        cv2.putText(frame, f"{k:<5}{v}", (FRAME_W - 164, 50 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HUD, 1)


def _draw_telemetry(frame, target, err_x, err_y, locked, infer_ok=True):
    _draw_panel(frame, 8, FRAME_H - 96, 290, FRAME_H - 8)
    cv2.putText(frame, "TELEMETRY", (16, FRAME_H - 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1)
    if target is None:
        msg, col = ("NO LOCK", COLOR_OTHER) if infer_ok else ("INFER ERR", COLOR_OTHER)
        cv2.putText(frame, msg, (16, FRAME_H - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        return
    state, state_col = ("LOCKED", COLOR_LOCK) if locked else ("TRACKING", COLOR_HUD)
    cv2.putText(frame, state, (140, FRAME_H - 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_col, 1)
    cv2.putText(frame, f"err   X {err_x:+5d}   Y {err_y:+5d}",
                (16, FRAME_H - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD, 1)
    cv2.putText(frame, f"pan   {pan:5.1f} deg",
                (16, FRAME_H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD, 1)
    cv2.putText(frame, f"tilt  {tilt:5.1f} deg",
                (16, FRAME_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD, 1)


def _detect(frame):
    res = model(frame, conf=conf_threshold, imgsz=INFER_SIZE,
                device=device, verbose=False)
    boxes = []
    for b in res[0].boxes:
        if target_class_idx is not None:
            if b.cls is None or int(b.cls[0]) != target_class_idx:
                continue
        boxes.append((int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                       int(b.xyxy[0][2]), int(b.xyxy[0][3]),
                       float(b.conf[0])))
    return boxes


def _pick_largest(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _slew(d):
    return _clamp(d, -MAX_STEP_DEG, MAX_STEP_DEG)


# ─── PUBLIC API ─────────────────────────────────────────────────────────────

def get_latest_jpeg():
    with _jpeg_lock:
        return _latest_jpeg


def inject_key(c):
    try:
        _key_queue.put_nowait(c.lower())
    except queue.Full:
        pass


def is_running():
    return _worker is not None and _worker.is_alive()


def set_tracking_active(on):
    """ACTIVE = drive the servos toward the target.
    PASSIVE = detect + draw bounding boxes only; turret stays put."""
    global tracking_active
    tracking_active = bool(on)


def get_tracking_active():
    return tracking_active


def stop():
    _stop_event.set()


def start(config):
    """Begin tracking in a background thread. Returns (ok, msg).
    config = {cam_index, capture_preset, model_path, target_class}"""
    global _worker, _stop_event, _key_queue
    global pan, tilt, capture_idx, conf_threshold, head_offset_frac
    global target_class_idx, device, model

    if is_running():
        return False, "tracker already running"

    # Load (or reload) model if needed.
    model_path = config.get("model_path", "yolov8n-face.pt")
    try:
        if model is None or getattr(model, "_loaded_path", None) != model_path:
            print(f"[tracker] loading model {model_path}")
            m = YOLO(model_path)
            m._loaded_path = model_path
            model = m
    except Exception as e:
        return False, f"model load failed: {e}"

    device = "mps" if (torch and torch.backends.mps.is_available()) else "cpu"

    # Resolve target class index if filtering.
    tc = (config.get("target_class") or "").strip()
    target_class_idx = None
    if tc and hasattr(model, "names"):
        names_iter = (model.names.items() if hasattr(model.names, "items")
                       else enumerate(model.names))
        for idx, name in names_iter:
            if str(name).lower() == tc.lower():
                target_class_idx = idx
                break
        if target_class_idx is None:
            print(f"[warn] class '{tc}' not found — tracking all classes")

    head_offset_frac = 0.5 if tc else HEAD_OFFSET_FRAC

    # Apply capture preset before opening the camera.
    cp = int(config.get("capture_preset", 0))
    capture_idx = cp if 0 <= cp < len(CAPTURE_PRESETS) else 0

    cam_index = int(config["cam_index"])

    # Reset per-launch state. Always begin PASSIVE — the operator must
    # explicitly arm tracking before any servo command goes out.
    pan = tilt = 90.0
    conf_threshold = 0.5
    set_tracking_active(False)

    # Drain any leftover keys from a prior session.
    while not _key_queue.empty():
        try: _key_queue.get_nowait()
        except queue.Empty: break

    _stop_event = threading.Event()
    _worker = threading.Thread(
        target=_run_loop, args=(cam_index,), daemon=True
    )
    _worker.start()
    send_servo(pan, tilt)
    return True, "started"


# ─── LOOP ───────────────────────────────────────────────────────────────────

def _run_loop(cam_index):
    global pan, tilt, capture_idx, conf_threshold, _latest_jpeg

    cap = open_camera(cam_index)
    if cap is None:
        print(f"[tracker] could not open camera {cam_index}")
        return

    # Warm up the model once so the first real frame isn't stalled.
    warm = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for _ in range(2):
        try:
            model(warm, imgsz=INFER_SIZE, device=device, verbose=False)
        except Exception:
            break

    prev_t  = time.time()
    fps     = 0.0
    cap_ms  = inf_ms = loop_ms = 0.0
    aim_x   = aim_y = None

    try:
      while not _stop_event.is_set():
        try:
            t_cap = time.time()
            ok, frame = cap.read()
            cap_ms = 0.9 * cap_ms + 0.1 * ((time.time() - t_cap) * 1000.0)
            if not ok:
                continue

            if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                frame = cv2.resize(frame, (FRAME_W, FRAME_H),
                                    interpolation=cv2.INTER_AREA)
            frame = cv2.flip(frame, 0)

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps     = 0.9 * fps     + 0.1 * (1.0 / dt)
                loop_ms = 0.9 * loop_ms + 0.1 * (dt * 1000.0)

            t_inf = time.time()
            infer_ok = True
            try:
                faces = _detect(frame)
            except Exception as e:
                print(f"[warn] inference failed ({type(e).__name__}: {e})")
                faces = []
                infer_ok = False
            inf_ms = 0.9 * inf_ms + 0.1 * ((time.time() - t_inf) * 1000.0)
            target = _pick_largest(faces)

            err_x = err_y = 0
            locked = False

            if target is not None:
                x1, y1, x2, y2, _ = target
                raw_x = (x1 + x2) // 2
                raw_y = int(y1 + head_offset_frac * (y2 - y1))
                if aim_x is None:
                    aim_x, aim_y = float(raw_x), float(raw_y)
                else:
                    aim_x = AIM_EMA_ALPHA * raw_x + (1 - AIM_EMA_ALPHA) * aim_x
                    aim_y = AIM_EMA_ALPHA * raw_y + (1 - AIM_EMA_ALPHA) * aim_y
                ix, iy = int(aim_x), int(aim_y)
                err_x  = ix - cx
                err_y  = iy - cy
                locked = (err_x * err_x + err_y * err_y) ** 0.5 <= DEADZONE_PX
                # Only update angles + write to serial when actively tracking.
                # In PASSIVE mode we still compute err / locked for the HUD
                # so the operator can see how the controller WOULD behave.
                if tracking_active and not locked:
                    d_tilt = _slew(KP * err_x) * (-1 if INVERT_TILT else 1)
                    d_pan  = _slew(KP * err_y) * (-1 if INVERT_PAN  else 1)
                    tilt   = _clamp(tilt + d_tilt, 0, 180)
                    pan    = _clamp(pan  + d_pan,  0, 180)
                    send_servo(pan, tilt)
            else:
                aim_x = aim_y = None

            for f in faces:
                if f is not target:
                    _draw_other(frame, f[:4], f[4])
            if target is not None:
                _draw_target(frame, target[:4], target[4])
                cv2.line(frame, (cx, cy), (int(aim_x), int(aim_y)), COLOR_TARGET, 1)
                cv2.circle(frame, (int(aim_x), int(aim_y)), 4, COLOR_TARGET, -1)

            _draw_crosshair(frame, locked)
            _draw_status(frame, fps, cam_index, ser is not None, len(faces),
                         cap_ms, inf_ms, loop_ms)
            _draw_controls(frame)
            _draw_telemetry(frame, target, err_x, err_y, locked, infer_ok)

            # Publish JPEG for the HTTP MJPEG endpoint to serve.
            ok_enc, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok_enc:
                with _jpeg_lock:
                    _latest_jpeg = buf.tobytes()

            # Pull queued keys (sent from the Electron renderer).
            while True:
                try:
                    k = _key_queue.get_nowait()
                except queue.Empty:
                    break
                if k == 'q':
                    _stop_event.set()
                elif k in ('+', '='):
                    conf_threshold = min(0.95, conf_threshold + 0.05)
                elif k in ('-', '_'):
                    conf_threshold = max(0.05, conf_threshold - 0.05)
                elif k == 'z':
                    capture_idx = (capture_idx + 1) % len(CAPTURE_PRESETS)
                    cap.release()
                    new_cap = open_camera(cam_index)
                    if new_cap is not None:
                        cap = new_cap
                        print(f"[zoom] capture now {cap_w}x{cap_h}")
                    else:
                        capture_idx = (capture_idx - 1) % len(CAPTURE_PRESETS)
                        cap = open_camera(cam_index)

        except Exception:
            print("\n[CRASH] iteration failed — full traceback below:")
            traceback.print_exc()
            print("[CRASH] continuing...\n")
            time.sleep(0.1)
    finally:
        try: cap.release()
        except Exception: pass
        # Clear the published frame so the next launch starts cleanly.
        with _jpeg_lock:
            _latest_jpeg = None
        print("[tracker] stopped")


# ─── ATEXIT ─────────────────────────────────────────────────────────────────

def _shutdown():
    stop()
    if ser is not None:
        try: ser.close()
        except Exception: pass
atexit.register(_shutdown)


# ─── STANDALONE (legacy direct-run path) ────────────────────────────────────

if __name__ == "__main__":
    # Legacy direct-run: assemble a config from env vars (set by the old
    # launcher.py) and start the tracker, then block on a cv2.imshow window.
    cfg = {
        "cam_index":      int(os.environ.get("TT_CAM_INDEX", "0")),
        "capture_preset": int(os.environ.get("TT_CAPTURE_PRESET", "0")),
        "model_path":     os.environ.get("TT_MODEL", "yolov8n-face.pt"),
        "target_class":   os.environ.get("TT_TARGET_CLASS", ""),
    }
    ok, msg = start(cfg)
    if not ok:
        raise SystemExit(msg)
    cv2.namedWindow("Turret Tracking")
    while is_running():
        jpeg = get_latest_jpeg()
        if jpeg is not None:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            cv2.imshow("Turret Tracking", img)
        k = cv2.waitKey(1) & 0xFF
        if k != 255:
            if k in (ord('q'), ord('Q')): inject_key('q')
            elif k in (ord('+'), ord('=')): inject_key('+')
            elif k in (ord('-'), ord('_')): inject_key('-')
            elif k in (ord('z'), ord('Z')): inject_key('z')
    cv2.destroyAllWindows()
