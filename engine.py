from PySide6.QtCore import QThread, Signal
import cv2
import mediapipe as mp
import time
import numpy as np
import math
from collections import deque, Counter

from inference import predict_intent, predict_gesture_from_features
from config_loader import load_config

# ---------------- CONFIG ----------------
cfg = load_config()
SMOOTHING_WINDOW = cfg["SMOOTHING_WINDOW"]
ON_CONF = cfg["ON_THRES"]
OFF_CONF = cfg["OFF_THRES"]
PADDING_RATIO = 0.35
COOLDOWN_SECONDS = 1.0

MP_SKIP = 2   # MediaPipe runs every N frames


class EngineWorker(QThread):
    gesture_signal = Signal(str)
    status_signal = Signal(str)

    def __init__(self, cam_number, orientation):
        super().__init__()
        self.cam_number = cam_number
        self.orientation = orientation
        self.running = True

        self.label_window = deque(maxlen=SMOOTHING_WINDOW)
        self.last_trigger_time = 0.0

        self.frame_id = 0
        self.last_landmarks = None
        self.prev_state = "NOTHING"   # FSM state

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    # -------- Majority vote --------
    def get_smoothed_label(self):
        return Counter(self.label_window).most_common(1)[0][0]

    # -------- Crop hand with padding --------
    def extract_hand_crop(self, frame, hand_landmarks):
        h, w, _ = frame.shape
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]

        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))

        px = int((x2 - x1) * PADDING_RATIO)
        py = int((y2 - y1) * PADDING_RATIO)

        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)

        if x2 <= x1 or y2 <= y1:
            return None, None

        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    # -------- Landmark → Relative Features --------
    def landmarks_to_features(self, hl, bbox, frame_shape):
        h, w, _ = frame_shape
        x1, y1, x2, y2 = bbox

        wrist = hl.landmark[0]
        middle = hl.landmark[9]

        wrist_px = (wrist.x * w - x1) / (x2 - x1)
        wrist_py = (wrist.y * h - y1) / (y2 - y1)

        middle_px = (middle.x * w - x1) / (x2 - x1)
        middle_py = (middle.y * h - y1) / (y2 - y1)

        palm_size = math.dist((wrist_px, wrist_py), (middle_px, middle_py))
        if palm_size < 0.015:
            return None

        features = []
        for p in hl.landmark:
            px = (p.x * w - x1) / (x2 - x1)
            py = (p.y * h - y1) / (y2 - y1)
            features.extend([
                (px - wrist_px) / palm_size,
                (py - wrist_py) / palm_size,
                (p.z - wrist.z) / palm_size
            ])

        return np.array(features, dtype=np.float32)

    # -------- Main Loop --------
    def run(self):
        self.status_signal.emit("Starting engine...")
        cap = cv2.VideoCapture(self.cam_number)

        if not cap.isOpened():
            self.status_signal.emit("Camera error")
            return

        self.status_signal.emit("Running")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            self.frame_id += 1
            frame = cv2.flip(frame, 1)

            width, height = 640, 480
            roi_width, roi_height = width - 20, height - 20

            if self.orientation == "Portrait":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                roi_width, roi_height = roi_height, roi_width

            cv2.rectangle(frame, (20, 20), (roi_width, roi_height), (0, 255, 0), 2)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---- MediaPipe frame skipping ----
            if self.frame_id % MP_SKIP == 0 or self.last_landmarks is None:
                results = self.hands.process(rgb)
                self.last_landmarks = results.multi_hand_landmarks
            else:
                results = None

            landmarks_to_use = (
                results.multi_hand_landmarks
                if results and results.multi_hand_landmarks
                else self.last_landmarks
            )

            # ---------------- FRAME-LEVEL INTENT AGGREGATION ----------------
            frame_has_intentional = False
            frame_intentional_conf = 0.0
            global_conf = 0.0

            if landmarks_to_use:
                for hl in landmarks_to_use:
                    crop, bbox = self.extract_hand_crop(frame, hl)
                    if crop is None:
                        continue

                    x1, y1, x2, y2 = bbox
                    if x1 < 20 or x2 > roi_width or y1 < 20 or y2 > roi_height:
                        continue

                    intent, i_conf = predict_intent(crop)
                    global_conf = max(global_conf, i_conf)

                    if intent == "intentional":
                        frame_has_intentional = True
                        frame_intentional_conf = max(frame_intentional_conf, i_conf)

            # ---------------- HYSTERESIS FSM ----------------
            if self.prev_state == "NOTHING":
                if frame_has_intentional and frame_intentional_conf >= ON_CONF:
                    self.prev_state = "INTENTIONAL"

            elif self.prev_state == "INTENTIONAL":
                if (not frame_has_intentional) or frame_intentional_conf < OFF_CONF:
                    self.prev_state = "NOTHING"

            # ---------------- TREE (ONLY IF INTENTIONAL) ----------------
            best_gesture = "unknown"
            best_gesture_conf = 0.0

            if self.prev_state == "INTENTIONAL" and landmarks_to_use:
                for hl in landmarks_to_use:
                    crop, bbox = self.extract_hand_crop(frame, hl)
                    if crop is None:
                        continue

                    features = self.landmarks_to_features(hl, bbox, frame.shape)
                    if features is None:
                        continue

                    gesture, g_conf = predict_gesture_from_features(features)
                    if g_conf > best_gesture_conf:
                        best_gesture = gesture
                        best_gesture_conf = g_conf

                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{gesture.upper()} ({g_conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            # ---------------- UI ----------------
            color = (0, 255, 0) if self.prev_state == "INTENTIONAL" else (0, 0, 255)
            cv2.putText(
                frame,
                f"INTENT: {self.prev_state} ({global_conf:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )

            # ---------------- SMOOTHING ----------------
            if self.prev_state == "INTENTIONAL":
                self.label_window.append(best_gesture)
            else:
                self.label_window.clear()

            if len(self.label_window) == SMOOTHING_WINDOW:
                now = time.time()
                if (now - self.last_trigger_time) >= COOLDOWN_SECONDS:
                    self.gesture_signal.emit(self.get_smoothed_label())
                    self.last_trigger_time = now
                self.label_window.clear()

            cv2.imshow("Gesture Engine", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.status_signal.emit("Stopped")

    def stop(self):
        self.running = False
