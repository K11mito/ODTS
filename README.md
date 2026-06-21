# ODTS — Open-source Detection and Tracking System

Closed-loop tracking turret.  A USB webcam streams to a YOLOv8 detector;
a P-controller drives servo angles over USB serial to an Arduino, which
steers the pan/tilt rig to keep the target centred on the crosshair.

## Downloads

- **Tracking software (this repo):** <https://github.com/K11mito/ODTS>
- **Arduino IDE** (to flash the Arduino firmware that receives `<pan,tilt,0>` commands and drives the servos): <https://www.arduino.cc/en/software>

## How it works

```
Camera ──▶ YOLOv8 ──▶ P-controller ──▶  <pan,tilt,0>\n  ──▶ Arduino ──▶ servos
                          ▲                                                │
                          └──────────── HUD + MJPEG ◀── Electron UI ◀──────┘
```

- An Electron app spawns a Python backend, which serves the UI and the live
  MJPEG video stream inside the same window — no separate cv2 window, no
  browser tab.
- The Python backend runs detection in a thread, encodes JPEGs into a
  one-slot buffer, and streams them to the Electron renderer.
- Camera reads happen on a dedicated thread so a glitchy USB webcam can't
  freeze the tracker — it'll show NO SIGNAL instead.

## Setup

### 1. Python dependencies
```
python -m venv .venv
.venv/bin/pip install opencv-python ultralytics pyserial numpy
```

### 2. Electron dependencies
```
npm install
```

### 3. Arduino firmware
Install the Arduino IDE from the link above. Flash an Arduino (Uno / Nano)
with a sketch that parses `<pan,tilt,0>\n` commands over serial at 115200
baud and writes the angles to two servo objects (pan + tilt).

Adjust the serial port at the top of `turret_track.py` if your Arduino
enumerates as something other than `/dev/cu.usbserial-110`.

## Run

```
npm start
```

You'll see a launcher window with detected camera thumbnails on the right
and target / capture-resolution options on the left.  Click a camera card,
pick a target (Face / Person / Water bottle / Cup / Cell phone / Book),
then hit **LAUNCH**.

## Live tracker controls

| Control | Effect |
|---|---|
| **TRACKING [PASSIVE / ACTIVE]** | ACTIVE arms the servos; PASSIVE detects + draws boxes only (turret stays still) |
| **CONF + / −** | Raise / lower detection confidence threshold |
| **ZOOM** | Cycle capture resolution: 480p → 720p → 1080p (wider native FOV on most webcams) |
| **CHANGE CAM** | Stop tracking and return to the camera picker |
| **STOP** | End tracking |

## Files

| File | Role |
|---|---|
| `main.js` | Electron main process — spawns the Python backend and loads the UI |
| `server.py` | HTTP backend — camera probe, MJPEG stream, JSON control endpoints |
| `turret_track.py` | Tracker module — detection, P-controller, serial output, threading |
| `index.html` | Launcher + live tracker UI in one page |
| `package.json` | Electron app config |
| `yolov8n-face.pt` | Face detection weights (other models auto-download on first use) |
| `face_detect_test.py` | Original face-detection prototype, kept for reference |
