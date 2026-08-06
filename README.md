# Facial Emotion Detection using Deep Learning

## 📌 Project Overview

This project is a real-time Facial Emotion Detection System built using TensorFlow, Keras, and OpenCV. It detects human faces from images or a webcam and predicts one of seven facial emotions using a Convolutional Neural Network (CNN).

---

## 🎯 Features

- Detects faces using OpenCV
- Predicts 7 facial emotions
- Real-time webcam emotion detection
- Image-based emotion prediction
- Trained CNN model
- Displays prediction confidence

---

## 😊 Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Dataset

FER-2013 Dataset

- Training Images: 28,709
- Test Images: 7,178
- Image Size: 48×48 (Grayscale)

---

## 🚀 How to Run

### Clone the repository

```bash
git clone <repository-url>
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python src/train.py
```

### Predict from an image

```bash
python src/predict.py
```

### Real-time webcam detection

```bash
python src/webcam.py
```

---

## 📊 Model Performance

- Training Accuracy: ~70%
- Validation Accuracy: ~58%

---

## 🔮 Future Improvements

- Data Augmentation
- Transfer Learning (MobileNetV2/EfficientNet)
- Streamlit Web Application
- Better Face Detection

---

## 👩‍💻 Author

Ahalya Ojha