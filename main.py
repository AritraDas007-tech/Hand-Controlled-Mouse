import os
import sys
import time
import math
import tempfile
import threading
import subprocess
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import cv2
import mediapipe as mp
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# --- Config & Thresholds ---
FRAME_W, FRAME_H = 640, 480
BOX_X1, BOX_X2 = int(FRAME_W * 0.2), int(FRAME_W * 0.8)
BOX_Y1, BOX_Y2 = int(FRAME_H * 0.2), int(FRAME_H * 0.8)

PINCH_PRESS = 0.28
PINCH_RELEASE = 0.40
FREEZE_THRESHOLD = 0.55
CLICK_COOLDOWN = 0.15
DRAG_DELAY = 0.40
DOUBLE_CLICK_WINDOW = 0.40
PAUSE_HOLD_TIME = 1.0
SCROLL_DEADZONE = 4.0
SCROLL_SENSITIVITY = 3.0
SCROLL_UNIT = 120 if sys.platform == "win32" else 1


class OneEuroFilter:
    def __init__(self, min_cutoff=1.2, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = t - self.t_prev
        if dt <= 0.0:
            return self.x_prev
        a_d = self._alpha(dt, self.d_cutoff)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def _alpha(self, dt, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class CameraStream:
    def __init__(self, src=0, width=FRAME_W, height=FRAME_H):
        if sys.platform == "win32" and isinstance(src, int):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(src)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if not self.cap.isOpened() or not self.grabbed:
            return False
        self.started = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed:
                break
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.started = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()


def extract_features(lm, w=FRAME_W, h=FRAME_H):
    # 3D hand size: Wrist (0) to Middle MCP (9)
    hand_size_3d = math.sqrt(
        (lm[0].x - lm[9].x) ** 2 +
        (lm[0].y - lm[9].y) ** 2 +
        (lm[0].z - lm[9].z) ** 2
    )
    hand_size_3d = max(hand_size_3d, 1e-4)

    # 3D Pinch Distances normalized by hand_size_3d
    d_thumb_index = math.sqrt(
        (lm[4].x - lm[8].x) ** 2 +
        (lm[4].y - lm[8].y) ** 2 +
        (lm[4].z - lm[8].z) ** 2
    ) / hand_size_3d

    d_thumb_middle = math.sqrt(
        (lm[4].x - lm[12].x) ** 2 +
        (lm[4].y - lm[12].y) ** 2 +
        (lm[4].z - lm[12].z) ** 2
    ) / hand_size_3d

    # Cursor tracking reference: Index MCP (5) blended with palm center
    palm_cx = (lm[0].x + lm[5].x + lm[9].x + lm[17].x) * 0.25
    palm_cy = (lm[0].y + lm[5].y + lm[9].y + lm[17].y) * 0.25
    cursor_ref_x = 0.5 * lm[5].x + 0.5 * palm_cx
    cursor_ref_y = 0.5 * lm[5].y + 0.5 * palm_cy

    def is_up(tip, pip):
        up_y = lm[tip].y < lm[pip].y
        further = math.hypot((lm[tip].x - lm[0].x) * w, (lm[tip].y - lm[0].y) * h) > \
                  math.hypot((lm[pip].x - lm[0].x) * w, (lm[pip].y - lm[0].y) * h)
        return up_y and further

    index_up = is_up(8, 6)
    middle_up = is_up(12, 10)
    ring_up = is_up(16, 14)
    pinky_up = is_up(20, 18)
    is_fist = not index_up and not middle_up and not ring_up and not pinky_up

    return {
        "hand_size_3d": hand_size_3d,
        "d_thumb_index": d_thumb_index,
        "d_thumb_middle": d_thumb_middle,
        "cursor_ref": (cursor_ref_x * w, cursor_ref_y * h),
        "index_up": index_up,
        "middle_up": middle_up,
        "ring_up": ring_up,
        "pinky_up": pinky_up,
        "is_fist": is_fist,
        "hand_y": (lm[8].y + lm[12].y) * 0.5 * h,
    }


class GestureStateMachine:
    def __init__(self):
        self.paused = False
        self.fist_start_time = None
        self.fist_latched = False

        self.active_pinch = None  # None, "LEFT", or "RIGHT"
        self.left_press_frames = 0
        self.left_release_frames = 0
        self.right_press_frames = 0
        self.right_release_frames = 0

        self.pinch_start_time = 0.0
        self.is_dragging = False
        self.last_tap_release = 0.0
        self.last_right_click = 0.0

        self.prev_scroll_y = None

    def update(self, features, screen_pos, now):
        d_ti = features["d_thumb_index"]
        d_tm = features["d_thumb_middle"]
        min_pinch_dist = min(d_ti, d_tm)
        freeze_cursor = (min_pinch_dist < FREEZE_THRESHOLD) and (not self.is_dragging)

        # 1. Closed Fist -> Pause/Resume Toggle (hold 1s)
        if features["is_fist"]:
            if self.fist_start_time is None:
                self.fist_start_time = now
            elif (now - self.fist_start_time >= PAUSE_HOLD_TIME) and not self.fist_latched:
                self.paused = not self.paused
                self.fist_latched = True
                if self.is_dragging:
                    pyautogui.mouseUp(button="left")
                    self.is_dragging = False
            return ("PAUSED" if self.paused else "HOLDING FIST"), False, True
        else:
            self.fist_start_time = None
            self.fist_latched = False

        if self.paused:
            return "PAUSED", False, True

        # 2. Pinch Detection & Handling (2-frame confirmation, smaller distance only)
        if self.active_pinch is None:
            # Pick whichever distance is smaller, never both
            if (d_tm < d_ti) and features["index_up"]:
                self.left_press_frames = 0
                if d_tm < PINCH_PRESS:
                    self.right_press_frames += 1
                    if self.right_press_frames >= 2:
                        self.active_pinch = "RIGHT"
                        self.right_press_frames = 0
                        if (now - self.last_right_click) >= 0.30:
                            pyautogui.click(button="right")
                            self.last_right_click = now
                            return "RIGHT CLICK", True, True
                else:
                    self.right_press_frames = 0
            else:
                self.right_press_frames = 0
                if d_ti < PINCH_PRESS:
                    self.left_press_frames += 1
                    if self.left_press_frames >= 2:
                        self.active_pinch = "LEFT"
                        self.pinch_start_time = now
                        self.left_press_frames = 0
                else:
                    self.left_press_frames = 0

        # Active RIGHT pinch state
        if self.active_pinch == "RIGHT":
            if (d_tm > PINCH_RELEASE) or (not features["index_up"]):
                self.right_release_frames += 1
                if self.right_release_frames >= 2:
                    self.active_pinch = None
                    self.right_release_frames = 0
            else:
                self.right_release_frames = 0

            if self.active_pinch == "RIGHT":
                return "RIGHT CLICK", True, True

        # Active LEFT pinch state
        if self.active_pinch == "LEFT":
            if d_ti > PINCH_RELEASE:
                self.left_release_frames += 1
                if self.left_release_frames >= 2:
                    self.active_pinch = None
                    self.left_release_frames = 0
                    if self.is_dragging:
                        pyautogui.mouseUp(button="left")
                        self.is_dragging = False
                        return "DROP", False, False
                    else:
                        dt = now - self.pinch_start_time
                        if dt < 0.40:
                            time_since_last_tap = now - self.last_tap_release
                            pyautogui.click(button="left")
                            if CLICK_COOLDOWN <= time_since_last_tap <= DOUBLE_CLICK_WINDOW:
                                self.last_tap_release = 0.0
                                return "DOUBLE CLICK", False, False
                            else:
                                self.last_tap_release = now
                                return "LEFT CLICK", False, False
            else:
                self.left_release_frames = 0

            if self.active_pinch == "LEFT":
                if not self.is_dragging and (now - self.pinch_start_time >= DRAG_DELAY):
                    self.is_dragging = True
                    pyautogui.mouseDown(button="left")

                if self.is_dragging:
                    pyautogui.moveTo(screen_pos[0], screen_pos[1], duration=0)
                    return "DRAG", True, False
                return "PINCH", True, True

        # 3. Scroll Mode (Index + Middle Up)
        if features["index_up"] and features["middle_up"] and not features["ring_up"] and not features["pinky_up"]:
            hand_y = features["hand_y"]
            if self.prev_scroll_y is not None:
                dy = hand_y - self.prev_scroll_y
                if abs(dy) >= SCROLL_DEADZONE:
                    scroll_amt = int(-dy * SCROLL_SENSITIVITY) * SCROLL_UNIT
                    if scroll_amt != 0:
                        pyautogui.scroll(scroll_amt)
                        self.prev_scroll_y = hand_y
            else:
                self.prev_scroll_y = hand_y
            return "SCROLL", False, False
        else:
            self.prev_scroll_y = None

        # 4. Move Mode (Index Up Only)
        if features["index_up"] and not features["middle_up"] and not features["ring_up"] and not features["pinky_up"]:
            if not freeze_cursor:
                pyautogui.moveTo(screen_pos[0], screen_pos[1], duration=0)
                return "MOVE", False, False
            return "FREEZE", False, True

        return "IDLE", False, freeze_cursor

    def reset_scroll(self):
        self.prev_scroll_y = None

    def cleanup(self):
        if self.is_dragging:
            pyautogui.mouseUp(button="left")
            self.is_dragging = False


def check_and_ensure_mediapipe():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return True
    py311 = r"C:\Users\aritr\AppData\Local\Programs\Python\Python311\python.exe"
    if os.path.exists(py311) and sys.executable != py311:
        ret = subprocess.call([py311, __file__] + sys.argv[1:])
        sys.exit(ret)
    print("Error: mediapipe.solutions.hands is not available in Python " + sys.version.split()[0], file=sys.stderr)
    print("Please run with Python 3.10-3.12 and mediapipe==0.10.14 as specified in requirements.txt.", file=sys.stderr)
    sys.exit(1)


def main():
    check_and_ensure_mediapipe()

    src = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    stream = CameraStream(src=src, width=FRAME_W, height=FRAME_H)
    if not stream.start():
        print(f"Error: Could not open camera (index {src}). Please check your webcam connection.", file=sys.stderr)
        return 1

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    screen_w, screen_h = pyautogui.size()
    fx, fy = OneEuroFilter(), OneEuroFilter()
    sm = GestureStateMachine()

    fps = 60.0
    prev_time = time.time()
    had_hand = False

    try:
        while True:
            t0 = time.time()
            ok, frame = stream.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            now = time.time()
            label = "NO HAND"
            is_pinched_visual = False
            d_ti_debug, d_tm_debug = 0.0, 0.0
            freeze_debug = False

            if results.multi_hand_landmarks:
                if not had_hand:
                    fx.reset()
                    fy.reset()
                    had_hand = True

                target_idx = 0
                if results.multi_handedness:
                    for i, handedness in enumerate(results.multi_handedness):
                        if handedness.classification[0].label == "Right":
                            target_idx = i
                            break
                lm = results.multi_hand_landmarks[target_idx]
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                feat = extract_features(lm.landmark, FRAME_W, FRAME_H)
                d_ti_debug = feat["d_thumb_index"]
                d_tm_debug = feat["d_thumb_middle"]

                # Cursor tracking reference: Index MCP blended with palm center
                rx, ry = feat["cursor_ref"]
                cv2.circle(frame, (int(rx), int(ry)), 6, (255, 255, 0), -1)

                cx = max(BOX_X1, min(BOX_X2, rx))
                cy = max(BOX_Y1, min(BOX_Y2, ry))
                nx = (cx - BOX_X1) / (BOX_X2 - BOX_X1)
                ny = (cy - BOX_Y1) / (BOX_Y2 - BOX_Y1)

                raw_sx = nx * (screen_w - 1)
                raw_sy = ny * (screen_h - 1)
                sx = int(max(0, min(screen_w - 1, fx(raw_sx, now))))
                sy = int(max(0, min(screen_h - 1, fy(raw_sy, now))))

                label, is_pinched_visual, freeze_debug = sm.update(feat, (sx, sy), now)
            else:
                had_hand = False
                sm.reset_scroll()

            # UI Rendering: Active Box
            box_color = (0, 0, 255) if is_pinched_visual else (0, 255, 0)
            cv2.rectangle(frame, (BOX_X1, BOX_Y1), (BOX_X2, BOX_Y2), box_color, 2)

            dt = max(1e-4, now - prev_time)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = now

            # Top UI
            cv2.putText(frame, f"Gesture: {label}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            # Debug Overlay (Normalized distances live)
            ti_col = (0, 255, 0) if d_ti_debug < PINCH_PRESS else ((0, 255, 255) if d_ti_debug < FREEZE_THRESHOLD else (200, 200, 200))
            tm_col = (0, 255, 0) if d_tm_debug < PINCH_PRESS else ((0, 255, 255) if d_tm_debug < FREEZE_THRESHOLD else (200, 200, 200))
            cv2.putText(frame, f"T-Index 3D: {d_ti_debug:.3f} (P<{PINCH_PRESS}, R>{PINCH_RELEASE})", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ti_col, 1)
            cv2.putText(frame, f"T-Middle 3D: {d_tm_debug:.3f} (Req Idx Up)", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.52, tm_col, 1)
            cv2.putText(frame, f"Cursor: {'FROZEN (<0.55)' if freeze_debug else 'TRACKING'}", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255) if freeze_debug else (0, 255, 0), 1)

            cv2.imshow("Hand Gesture Mouse", frame)

            # Exit on 'q' or window close
            elapsed = time.time() - t0
            wait_ms = max(1, int((1.0 / 60.0 - elapsed) * 1000))
            if (cv2.waitKey(wait_ms) & 0xFF == ord("q")) or (cv2.getWindowProperty("Hand Gesture Mouse", cv2.WND_PROP_VISIBLE) < 1):
                break
    finally:
        sm.cleanup()
        stream.stop()
        hands.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
