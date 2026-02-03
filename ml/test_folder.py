import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# ── Configuration ────────────────────────────────────────────────
MODEL_PATH      = 'pothole_detector.h5'          # or your improved model name
TEST_FOLDER     = 'test'                  # folder name inside ml/
IMG_HEIGHT      = 128
IMG_WIDTH       = 128
THRESHOLD       = 0.59                           # adjust as needed (0.25–0.45 range)

# ── Load model ───────────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully\n")

# ── Prediction function ──────────────────────────────────────────
def predict_single(img_path, threshold=THRESHOLD):
    try:
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prob = model.predict(img_array, verbose=0)[0][0]
        
        pred = "Pothole" if prob > threshold else "No pothole"
        if prob < 0.55 and prob > 0.45:
            pred = "Bumpy road"  # subtle hint for borderline cases
        return prob, pred
    except Exception as e:
        return None, f"Error: {str(e)}"

# ── Batch test folder ────────────────────────────────────────────
print(f"Scanning folder: {TEST_FOLDER}\n")

if not os.path.isdir(TEST_FOLDER):
    print(f"Error: Folder '{TEST_FOLDER}' not found in current directory.")
else:
    image_files = [f for f in os.listdir(TEST_FOLDER) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No images found in the folder.")
    else:
        print(f"Found {len(image_files)} images. Starting predictions...\n")
        
        results = []
        
        for idx, filename in enumerate(image_files, 1):
            full_path = os.path.join(TEST_FOLDER, filename)
            prob, prediction = predict_single(full_path)
            
            if prob is not None:
                status = f"{prediction} ({prob:.4f})"
                print(f"[{idx}/{len(image_files)}] {filename: <30} → {status}")
                results.append((filename, prob, prediction))
            else:
                print(f"[{idx}/{len(image_files)}] {filename: <30} → {prediction}")
        
        print("\n" + "="*60)
        print("Summary:")
        pothole_count = sum(1 for _, p, pred in results if pred == "Pothole")
        print(f"Total images tested : {len(results)}")
        print(f"Predicted as Pothole : {pothole_count} ({pothole_count/len(results)*100:.1f}%)")
        print(f"Average probability  : {np.mean([p for _, p, _ in results if p is not None]):.4f}")