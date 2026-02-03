# test_single_image_simple.py
# Just change the image path below and run the script

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ── Settings (change only if you trained with different size) ──
MODEL_FILE = "pothole_cnn_simple.h5"
IMG_HEIGHT = 128
IMG_WIDTH  = 128

# ── CHANGE THIS LINE every time you test a new image ───────────
image_path = "test2_pothole.jpg"
# examples:
# image_path = "new_dataset/test/India_000081.jpg"
# image_path = "new_dataset/test/15766.jpg"
# image_path = "test.jpg"
# image_path = "D:/projects/POTHole-DETECTION/ml/test3.jpg"

# ── Load model ─────────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_FILE)
print("Model loaded.")

# ── Load and prepare image ─────────────────────────────────────
img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)   # add batch dimension
x = x / 255.0                   # normalize

# ── Predict ────────────────────────────────────────────────────
pred = model.predict(x)[0][0]

print("\n" + "-" * 50)
print(f"Image: {image_path}")
print("-" * 50)

if pred > 0.5:
    print("Result:     Pothole")
    print(f"Confidence: {pred*100:.1f}%")
else:
    print("Result:     Normal road / No pothole")
    print(f"Confidence: {(1-pred)*100:.1f}%")

print("-" * 50 + "\n")