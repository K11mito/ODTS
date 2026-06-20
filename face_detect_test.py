import time
import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None

# Load YOLOv8n-face weights — file must sit next to this script (or full path).
model = YOLO("yolov8n-face.pt")

# Pin device/infer size to match the production tracker so timings are comparable.
DEVICE = "mps" if (torch and torch.backends.mps.is_available()) else "cpu"
INFER_SIZE = 320

# Capture/render dimensions; centre is recomputed once for the crosshair and error vectors.
FRAME_W, FRAME_H = 640, 480
cx, cy = FRAME_W // 2, FRAME_H // 2

# Live-tunable detection threshold — adjusted with +/-.
conf_threshold = 0.5

# Colour palette (BGR). Kept as constants so HUD elements stay visually consistent.
COLOR_TARGET = (80, 255, 120)     # tracked face
COLOR_OTHER = (70, 70, 230)       # ignored faces
COLOR_HUD = (240, 240, 240)       # neutral text
COLOR_ACCENT = (0, 200, 255)      # amber accents (titles, frame edges)
COLOR_PANEL = (15, 15, 15)        # dark panel fill
COLOR_CROSSHAIR = (0, 220, 255)


def try_open_camera(idx):
    # Attempt to open a camera and confirm it actually delivers a frame —
    # macOS sometimes "opens" a phantom device that never produces images.
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # serve the freshest frame, not a queued one
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def probe_cameras(max_idx=4):
    # Walk indices 0..max_idx and return the ones that yield a frame.
    # On macOS this enumerates built-in webcam, Continuity Camera (iPhone), and any USB cams.
    found = []
    for i in range(max_idx + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found


def draw_panel(frame, x1, y1, x2, y2, alpha=0.55):
    # Semi-transparent dark backing for HUD text — improves readability over busy backgrounds.
    # Blend only the panel sub-rectangle to avoid copying the whole frame each call.
    h, w = frame.shape[:2]
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(w, x2), min(h, y2)
    if rx2 > rx1 and ry2 > ry1:
        roi = frame[ry1:ry2, rx1:rx2]
        fill = np.empty_like(roi)
        fill[:] = COLOR_PANEL
        cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, dst=roi)
    # Thin amber border gives panels a defined edge against any scene.
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ACCENT, 1)


def draw_crosshair(frame):
    # Outer ring acts as a visual "deadzone" — once the target sits inside it the turret can stop slewing.
    cv2.circle(frame, (cx, cy), 30, COLOR_CROSSHAIR, 1)
    # Crosshair lines with a centre gap so the operator can see the actual centre pixel.
    cv2.line(frame, (cx - 25, cy), (cx - 8, cy), COLOR_CROSSHAIR, 1)
    cv2.line(frame, (cx + 8, cy), (cx + 25, cy), COLOR_CROSSHAIR, 1)
    cv2.line(frame, (cx, cy - 25), (cx, cy - 8), COLOR_CROSSHAIR, 1)
    cv2.line(frame, (cx, cy + 8), (cx, cy + 25), COLOR_CROSSHAIR, 1)
    cv2.circle(frame, (cx, cy), 2, COLOR_CROSSHAIR, -1)


def draw_target_brackets(frame, x1, y1, x2, y2):
    # L-shaped corner brackets — reads as a "lock" reticle rather than a plain rectangle.
    L = 18  # bracket arm length in pixels
    t = 2
    cv2.line(frame, (x1, y1), (x1 + L, y1), COLOR_TARGET, t)
    cv2.line(frame, (x1, y1), (x1, y1 + L), COLOR_TARGET, t)
    cv2.line(frame, (x2, y1), (x2 - L, y1), COLOR_TARGET, t)
    cv2.line(frame, (x2, y1), (x2, y1 + L), COLOR_TARGET, t)
    cv2.line(frame, (x1, y2), (x1 + L, y2), COLOR_TARGET, t)
    cv2.line(frame, (x1, y2), (x1, y2 - L), COLOR_TARGET, t)
    cv2.line(frame, (x2, y2), (x2 - L, y2), COLOR_TARGET, t)
    cv2.line(frame, (x2, y2), (x2, y2 - L), COLOR_TARGET, t)


def draw_target_label(frame, x1, y1, conf):
    # Filled label tag above the target — high-contrast text for the metric the operator cares most about.
    label = f"TARGET  {conf * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), COLOR_TARGET, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def draw_other_face(frame, box, conf):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OTHER, 1)
    cv2.putText(frame, f"{conf * 100:.0f}%", (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_OTHER, 1)


def draw_top_hud(frame, fps, cam_idx, cam_total, num_faces):
    # Left panel: status block.
    draw_panel(frame, 8, 8, 230, 86)
    cv2.putText(frame, "TURRET TRACKING TEST", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ACCENT, 1)
    cv2.putText(frame, f"Conf >= {conf_threshold:.2f}", (16, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD, 1)
    cv2.putText(frame, f"Cam {cam_idx}/{cam_total - 1}   {fps:5.1f} FPS   Faces {num_faces}",
                (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HUD, 1)

    # Right panel: controls cheatsheet — saves the operator from memorising shortcuts.
    draw_panel(frame, FRAME_W - 188, 8, FRAME_W - 8, 86)
    cv2.putText(frame, "CONTROLS", (FRAME_W - 180, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1)
    cv2.putText(frame, "Q     quit", (FRAME_W - 180, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HUD, 1)
    cv2.putText(frame, "+/-   conf threshold", (FRAME_W - 180, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HUD, 1)
    cv2.putText(frame, "C     next camera", (FRAME_W - 180, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HUD, 1)


def draw_bottom_hud(frame, target):
    draw_panel(frame, 8, FRAME_H - 96, 270, FRAME_H - 8)
    cv2.putText(frame, "TARGET TELEMETRY", (16, FRAME_H - 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ACCENT, 1)

    if target is None:
        # Distinct "no lock" state so the operator can tell threshold-too-high from no-faces-in-frame.
        cv2.putText(frame, "NO LOCK", (16, FRAME_H - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_OTHER, 2)
        return

    x1, y1, x2, y2, conf = target
    face_w = x2 - x1
    # Face centre — point the servo loop will drive toward the crosshair.
    fcx = (x1 + x2) // 2
    fcy = (y1 + y2) // 2
    # Error: face-centre minus frame-centre. +X = face right of centre, +Y = below centre.
    err_x = fcx - cx
    err_y = fcy - cy

    # Visualise the error vector so the numbers can be sanity-checked at a glance.
    cv2.line(frame, (cx, cy), (fcx, fcy), COLOR_TARGET, 1)
    cv2.circle(frame, (fcx, fcy), 4, COLOR_TARGET, -1)

    cv2.putText(frame, f"conf   {conf * 100:5.1f} %", (16, FRAME_H - 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD, 1)
    cv2.putText(frame, f"width  {face_w:4d} px", (16, FRAME_H - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD, 1)
    cv2.putText(frame, f"err    X {err_x:+5d}   Y {err_y:+5d}", (16, FRAME_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD, 1)


def detect_faces(frame):
    # verbose=False keeps ultralytics from spamming the terminal — all feedback stays on-screen.
    results = model(frame, conf=conf_threshold, imgsz=INFER_SIZE,
                    device=DEVICE, verbose=False)
    faces = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        faces.append((x1, y1, x2, y2, conf))
    return faces


def pick_largest(faces):
    # Largest-area face is the closest one; that's what the turret should track.
    if not faces:
        return None
    return max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))


# --- startup ---------------------------------------------------------------
# Probe up-front so 'C' can cycle through real, working cameras instead of guessing.
available = probe_cameras(4)
if not available:
    raise SystemExit("No cameras found — check OS camera permission for the terminal app.")

# Prefer a likely external/USB camera as default (typically non-zero index),
# then fall back to the first detected camera.
default_cam_idx = next((idx for idx in available if idx != 0), available[0])
# cam_pos indexes into `available`; available[cam_pos] is the actual OpenCV index.
cam_pos = available.index(default_cam_idx)
cap = try_open_camera(available[cam_pos])
if cap is None:
    raise SystemExit("Camera failed to open after probe.")

# Warm up the model so the first real frame isn't stalled by lazy init / graph build.
_warm = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
for _ in range(3):
    model(_warm, imgsz=INFER_SIZE, device=DEVICE, verbose=False)

# FPS state — exponential moving average so the readout doesn't jitter every frame.
prev_t = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    # A dropped frame shouldn't crash the loop; just try the next read.
    if not ok:
        continue
    # Flip vertically so the camera feed appears upside down.
    frame = cv2.flip(frame, 0)

    # Update FPS estimate using time between successful reads.
    now = time.time()
    dt = now - prev_t
    prev_t = now
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    faces = detect_faces(frame)
    target = pick_largest(faces)

    # Draw non-target faces first so the target overlay stays on top visually.
    for f in faces:
        if f is target:
            continue
        draw_other_face(frame, f[:4], f[4])

    if target is not None:
        x1, y1, x2, y2, conf = target
        draw_target_brackets(frame, x1, y1, x2, y2)
        draw_target_label(frame, x1, y1, conf)

    draw_crosshair(frame)
    draw_top_hud(frame, fps, available[cam_pos], len(available), len(faces))
    draw_bottom_hud(frame, target)

    cv2.imshow("YOLOv8 Face Detector Test", frame)

    # waitKey(1) gives the window ~1ms to render; & 0xFF normalises to a single byte for portable codes.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    # '=' is the unshifted key on most layouts where '+' lives — accept both so Shift isn't required.
    elif key == ord('+') or key == ord('='):
        conf_threshold = min(0.95, conf_threshold + 0.05)
    elif key == ord('-') or key == ord('_'):
        conf_threshold = max(0.05, conf_threshold - 0.05)
    elif key == ord('c') or key == ord('C'):
        # Cycle to next available camera; if the new one fails to open, restore the previous.
        if len(available) > 1:
            next_pos = (cam_pos + 1) % len(available)
            cap.release()
            new_cap = try_open_camera(available[next_pos])
            if new_cap is not None:
                cap = new_cap
                cam_pos = next_pos
            else:
                # Restore prior camera so we never end up with a dead cap object.
                cap = try_open_camera(available[cam_pos])

cap.release()
cv2.destroyAllWindows()
