# Just Dance Pose Detection Game

## 🕺 Project Overview

This project is an interactive computer vision game that uses AI and pose detection to create a fun, engaging fitness and dance experience. Players are challenged to perform specific body poses while the system tracks and scores their accuracy in real-time.

## 🌟 Features

- Real-time pose detection using YOLOv8 and MediaPipe
- Multiple pose challenges
- Automatic score tracking
- Dynamic pose selection
- Visual feedback on pose correctness

## 📋 Prerequisites

### Hardware Requirements
- Webcam
- Computer with moderate GPU capabilities

### Software Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- Ultralytics YOLO
- NumPy

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/BapJu/Just-Dance-Pose-Detection-Game.git
cd dance-pose-game
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 How to Play

1. Run the game:
```bash
python dance_game.py
```

2. Stand in front of your webcam
3. Try to match the displayed pose
4. Hold the correct pose for 1.5 seconds to score points
5. Press 'q' to quit the game

## 🧩 Poses Available

- Prayer Pose
- Hands to Left
- Hands to Right
- Hands on Hips
- Hands in Air
- Spread Arms

## 🔧 Customization

You can easily extend the game by:
- Adding more poses to `POSITIONS` list
- Modifying scoring mechanics
- Adjusting pose detection thresholds

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 👥 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/yolov5) for YOLO
- [MediaPipe](https://mediapipe.dev/) for pose estimation
- OpenCV for computer vision processing

## 🚧 Known Issues

- Requires good lighting conditions
- Accuracy depends on webcam quality
- Some poses might be challenging to detect precisely
