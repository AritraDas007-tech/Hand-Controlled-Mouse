# Hand-Controlled Virtual Mouse

A high-performance, jitter-free Python virtual mouse driven by real-time hand gestures using OpenCV, MediaPipe, and PyAutoGUI.

---

## Features & Highlights

- **Stable Cursor Tracking**: Tracks the **Index MCP (landmark 5)** blended with the **Palm Center**, avoiding fingertip twitching and drift during clicks.
- **Pre-Click Cursor Freeze**: Automatically freezes cursor position when fingers approach ($\text{distance} < 0.55$) to prevent slipping off small UI elements or buttons.
- **3D Normalized Pinch Detection**: Calculates 3D Euclidean distances normalized by hand scale (wrist-to-middle-MCP), invariant to distance from the webcam.
- **Two-Frame Confirmation with Hysteresis**: Press threshold ($< 0.28$) and release threshold ($> 0.40$) requiring 2 consecutive frames prevents single-frame noise glitches.
- **Mutual Exclusion**: Evaluates the smaller distance between thumb-index and thumb-middle pinches so left and right clicks never collide.
- **One Euro Filter Smoothing**: Adaptive frequency cutoff reduces jitter at slow speeds and eliminates cursor lag at high speeds.
- **Multi-Threaded Camera Pipeline**: Camera capture runs in a background thread at 640x480 resolution, decoupled from UI/processing and capped at 60 FPS.
- **Live Debug Overlay**: Shows real-time normalized distances, cursor freeze state, active gesture, and FPS for easy threshold tuning.

---

## Gestures (Right Hand, Mirrored Frame)

| Gesture | Hand Trigger | Action |
| :--- | :--- | :--- |
| **Move** | Index finger up, other fingers folded | Cursor tracks hand movement smoothly |
| **Left Click** | Thumb + Index pinch ($< 0.4\text{s}$ tap) | Fires left mouse click on release |
| **Double Click** | Two quick thumb + index pinches within $0.4\text{s}$ | Fires double click |
| **Drag & Drop** | Hold thumb + index pinch $> 0.4\text{s}$ | `mouseDown` held while moving; release pinch to drop (`mouseUp`) |
| **Right Click** | Thumb + Middle pinch with **Index extended** | Fires right click |
| **Scroll** | Index + Middle extended together (peace sign) | Vertical hand movement scrolls up/down |
| **Pause / Resume** | Closed fist held for $1.0\text{s}$ | Toggles virtual mouse active / paused state |
| **Quit** | Press `q` or close window | Graceful shutdown, releases mouse & camera |

---

## Tech Stack

- **Python**: 3.10 – 3.12 (Recommended: Python 3.11)
- **MediaPipe**: `==0.10.14` (pinned for `mp.solutions.hands` and `model_complexity=1`)
- **OpenCV**: `opencv-python>=4.8.0`
- **PyAutoGUI**: `pyautogui>=0.9.54`
- **NumPy**: `numpy>=1.24.0`

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AritraDas007-tech/Hand-Controlled-Mouse.git
   cd Hand-Controlled-Mouse
   ```

2. **Create and activate a virtual environment (Python 3.10 - 3.12):**
   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run the Virtual Mouse
```bash
python main.py
```
*Optional: Select camera index (e.g. external webcam at index 1):*
```bash
python main.py 1
```

### Run Unit Tests
A dedicated unit test suite verifies all gestures, state machine transitions, 2-frame confirmations, and the One Euro filter using synthetic 3D landmark data:
```bash
python test_gesture.py
```

---

## Live Debug Overlay & Threshold Tuning

The mirrored preview window displays a real-time HUD showing:
- `T-Index 3D`: Live thumb-to-index distance (Press $< 0.28$, Release $> 0.40$).
- `T-Middle 3D`: Live thumb-to-middle distance with Index Up requirement.
- `Cursor`: Displays `FROZEN (<0.55)` or `TRACKING`.

### 5-Line Tuning Guide
1. **Pinch Sensitivity (`PINCH_PRESS = 0.28`, `PINCH_RELEASE = 0.40`)**: Increase `PINCH_PRESS` (e.g. `0.32`) if pinches don't register; decrease if triggering prematurely.
2. **Pre-Click Freeze (`FREEZE_THRESHOLD = 0.55`)**: Increase (e.g. `0.60`) to freeze cursor earlier before pinching; lower if you want cursor tracking closer to contact.
3. **Smoothness vs Responsiveness (`min_cutoff = 1.2`, `beta = 0.05`)**: Lower `min_cutoff` (`0.8`) to reduce micro-jitters when hand is still; increase `beta` (`0.08`) for faster tracking.
4. **Active Box Bounds (`BOX_X1/X2`, `BOX_Y1/Y2` = 20% to 80%)**: Shrink to 25%–75% for shorter physical arm reach; expand to 15%–85% for wider reach.
5. **Scroll Sensitivity (`SCROLL_DEADZONE = 4.0`, `SCROLL_SENSITIVITY = 3.0`)**: Increase sensitivity for faster page scrolling; increase deadzone if scrolling triggers unintentionally.

---

## License

MIT License.
