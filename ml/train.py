import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# ── Parameters ────────────────────────────────────────
IMG_HEIGHT = 128
IMG_WIDTH  = 128
BATCH_SIZE = 32          # start with 32; reduce to 16/8 if OOM
EPOCHS     = 30          # you can increase or use early stopping
SEED       = 42

# Paths (adjust if needed)
TRAIN_DIR = "new_dataset/train"
TEST_DIR  = "new_dataset/test"

# ── Data Augmentation (helps a lot with small datasets) ──
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)   


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',          # binary → pothole (1) vs normal (0)
    shuffle=True,
    seed=SEED
)

validation_generator = test_datagen.flow_from_directory(   # we use test as val for simplicity
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False                 # important for clean evaluation
)

# Print class indices (should be {'Plain': 0, 'Pothole': 1} or similar)
print("Class indices:", train_generator.class_indices)

# ── Simple but effective CNN model ────────────────────
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')   # binary output
])

model.summary()

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ── Train ─────────────────────────────────────────────
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1
)

# ── Save model ────────────────────────────────────────
model.save("pothole_cnn_simple.h5")
print("Model saved as pothole_cnn_simple.h5")

# ── Evaluate on test set (accuracy & loss) ────────────
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

# Optional: Predict on few images (example)
# from tensorflow.keras.preprocessing import image
# img = image.load_img("new_dataset/test/Pothole/some.jpg", target_size=(128,128))
# img_array = image.img_to_array(img) / 255.0
# img_array = np.expand_dims(img_array, axis=0)
# pred = model.predict(img_array)[0][0]
# print("Pothole probability:", pred)