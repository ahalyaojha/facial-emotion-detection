import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("models/emotion_model.keras")

# Emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Load image
image_path = "images/test.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize to 48x48
gray = cv2.resize(gray, (48, 48))

# Normalize
gray = gray.astype("float32") / 255.0

# Reshape for CNN
gray = np.expand_dims(gray, axis=-1)
gray = np.expand_dims(gray, axis=0)

# Predict
prediction = model.predict(gray)

emotion = emotion_labels[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("Predicted Emotion:", emotion)
print(f"Confidence: {confidence:.2f}%")