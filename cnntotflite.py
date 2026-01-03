import tensorflow as tf
from config_loader import load_config

# ---------------- CONFIG ----------------
cfg = load_config()
CNN_PATH = cfg["MODEL_PATH_CNN"]
IMG_SIZE = cfg["IMG_SIZE"]

TFLITE_PATH = "cnn_intent_fp32.tflite"

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(CNN_PATH)
model.trainable = False

# ---------------- CREATE CONCRETE FUNCTION ----------------
@tf.function(input_signature=[
    tf.TensorSpec(
        shape=[1, IMG_SIZE, IMG_SIZE, 3],
        dtype=tf.float32
    )
])
def model_fn(x):
    return model(x)

concrete_func = model_fn.get_concrete_function()

# ---------------- CONVERT ----------------
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func]
)

# IMPORTANT FLAGS (fix MLIR issues)
converter.experimental_enable_resource_variables = False
converter.optimizations = []  # FP32, no quantization

tflite_model = converter.convert()

# ---------------- SAVE ----------------
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("✅ TFLite conversion successful:", TFLITE_PATH)
