# Hand and Face Detection using TensorFlow and Keras

This project demonstrates a computer vision system that integrates **Hand Detection**, **Face Detection**, and **Emotion Recognition** using **Python**, **TensorFlow**, and **Keras**. It utilizes live camera input to detect facial expressions and count raised fingers using landmark estimation.

> ✅ Python Version: 3.9 – 3.12  
> ✅ Requires: TensorFlow, Keras, OpenCV, Mediapipe, Numpy

---

## Features

### 🖐 Hand Detection and Finger Counting
- Detects hand landmarks using **Mediapipe**'s `Hands` model.
- Represents 21 key **vertices** (landmark points) of the hand connected by **lines** (edges) for a skeletal hand structure.
- Determines the number of extended fingers and displays it in real time.
- Example: Displaying "2" with an open index and middle finger results in a "Peace ✌️" sign.

📷 Example Output:  
<div align="center">
  <img src="https://i.postimg.cc/65DLGgFx/Whats-App-Image-2025-05-17-at-04-42-27.jpg" alt="Demo Image" style="width: 300px; height: auto;" />
</div>

---

### 😊 Emotion Detection from Face
- Detects faces using Haar Cascade Classifier.
- Extracts the facial region, converts to grayscale, resizes to `64x64`, normalizes pixels, and passes through a pre-trained CNN model using Keras.
- Supported emotions:
  - `Angry`, `Disgusted`, `Feared`, `Happy`, `Sad`, `Surprise`, `Neutral`

## 🧩 Installation

### ✅ Requirements

- Python: `3.9`, `3.10`, `3.11`, atau `3.12` (as per May 2025)
- TensorFlow dan other dependencies:

Install TensorFlow: [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip)

Install all packages using pip:

```bash
pip install tensorflow keras opencv-python numpy collections mediapipe
```

## ▶️ How to Run
1. Clone this repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
2. Run the main script:
```bash
python main.py
```
3. Output:
- A webcam window will appear.
- Show your hand gesture — the number of fingers will be detected and visualized with lines and dots (landmark graph).
- Your face will be analyzed in real-time and emotions will be classified from the following set:
```python
emotions = ("Angry", "Disgusted", "Feared", "Happy", "Sad", "Surprise", "Neutral")
```
Internally, it processes detected face using:
```python
detected_face = frame[int(cy_min):int(cy_max), int(cx_min):int(cx_max)]
detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
detected_face = cv2.resize(detected_face, (64, 64))

frame_pixels = img_keras.img_to_array(detected_face)
frame_pixels = np.expand_dims(frame_pixels, axis=0)
frame_pixels /= 255
emotion = model.predict(frame_pixels)[0]
Q.append(emotion)
```

