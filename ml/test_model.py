import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model("pothole_cnn_simple.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    "new_dataset/test",
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print("Evaluating final model on test set...")
loss, acc = model.evaluate(test_gen)
print(f"Final Test Accuracy: {acc*100:.2f}%")

# Confusion matrix & per-class performance (optional but useful)
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = (model.predict(test_gen) > 0.5).astype(int)
y_true = test_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pothole']))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Pothole'],
            yticklabels=['Normal', 'Pothole'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()