import warnings

warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype() is deprecated"
)
import cv2
import numpy as np
import tensorflow as tf
import joblib
import math

from config_loader import load_config

# ---------------- CONFIG ----------------
cfg = load_config()

CNN_PATH  = cfg["MODEL_PATH_CNN"]   # .tflite
TREE_PATH = cfg["MODEL_PATH_TREE"]
IMG_SIZE  = cfg["IMG_SIZE"]
CNN_THRESHOLD = cfg["CONF_THRESHOLD"]

GESTURE_LABELS = ["closed", "four", "open", "two"]

# ---------------- LOAD TFLITE CNN ----------------
interpreter = tf.lite.Interpreter(model_path=CNN_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- LOAD TREE ----------------
bundle = joblib.load(TREE_PATH)
tree   = bundle["model"]
scaler = bundle["scaler"]

# ======================================================
# CNN INTENT ONLY
# ======================================================
def predict_intent(img):
    cnn_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cnn_img = cv2.resize(cnn_img, (IMG_SIZE, IMG_SIZE))
    cnn_img = cnn_img.astype(np.float32) / 255.0
    cnn_img = np.expand_dims(cnn_img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], cnn_img)
    interpreter.invoke()

    p_nothing = float(
        interpreter.get_tensor(output_details[0]["index"])[0][0]
    )

    intent_conf = 1.0 - p_nothing

    if intent_conf < CNN_THRESHOLD:
        return "nothing", 1.0 - intent_conf

    return "intentional", intent_conf


# ======================================================
# GEOMETRIC FINGER COUNT (MediaPipe landmarks)
# ======================================================
def count_extended_fingers(hand_landmarks):
    """
    Counts extended fingers using tip vs PIP joint geometry.
    Thumb excluded (more robust).
    """
    # finger tips and PIP joints (index → pinky)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    count = 0
    for tip, pip in zip(tips, pips):
        # y-axis: smaller is higher (finger up)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            count += 1

    return count


# ======================================================
# TREE PREDICTION + GEOMETRIC OVERRIDE
# ======================================================
def predict_gesture_from_features(features, hand_landmarks=None):
    if features is None:
        return "unknown", 0.0

    if np.any(np.isnan(features)) or np.any(np.abs(features) > 5):
        return "unknown", 0.0

    X = scaler.transform(features.reshape(1, -1))
    pred = int(tree.predict(X)[0])
    gesture_label = GESTURE_LABELS[pred]

    if hasattr(tree, "predict_proba"):
        gesture_conf = float(np.max(tree.predict_proba(X)))
    else:
        gesture_conf = 0.9

    # ---------------- GEOMETRIC SANITY CHECK ----------------
    if hand_landmarks is not None:
        finger_count = count_extended_fingers(hand_landmarks)
        print(finger_count)

        # Fix: OPEN misclassified as TWO
        if gesture_label == "two" and finger_count >= 4:
            gesture_label = "open"
            gesture_conf *= 0.8   # slightly penalize

        # Optional: TWO misclassified as OPEN
        elif gesture_label == "open" and finger_count == 2:
            gesture_label = "two"
            gesture_conf *= 0.8

        # Optional: CLOSED sanity
        elif gesture_label == "closed" and finger_count >= 3:
            gesture_label = "open"
            gesture_conf *= 0.7

        if finger_count == 3:
            gesture_label = "three"
            gesture_conf = 0.7
        elif finger_count == 1:
            gesture_label = "one"
            gesture_conf = 0.7

    return gesture_label, gesture_conf
