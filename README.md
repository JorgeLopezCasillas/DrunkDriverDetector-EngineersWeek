
# DrunkDriverDetector-EngineersWeek

## Project Overview
This project aims to create an application that helps detect drunk drivers using computer vision techniques. The system uses facial recognition and eye status detection to monitor the driver's alertness in real time.

## Features
- Real-time facial landmark detection using webcam
- Eye status detection (open/closed) using Eye Aspect Ratio (EAR)
- Visual feedback on screen indicating if eyes are open or closed

## Setup Instructions
1. Clone this repository:
	```bash
	git clone https://github.com/JorgeLopezCasillas/DrunkDriverDetector-EngineersWeek.git
	```
2. Install required Python packages:
	```bash
	pip install opencv-python dlib imutils numpy
	```
3. Download the facial landmark predictor model:
	- [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
	- Extract and place `shape_predictor_68_face_landmarks.dat` in the `FacialRecognition/` directory.

## Usage
Run the facial recognition script:
```bash
cd FacialRecognition
python FacialRecognition.py
```
The application will open your webcam and display facial landmarks. It will show "Eyes Open" or "Eyes Closed" on the screen based on your eye status. Press `q`, `Q`, or `Esc` to exit.

## Dependencies
- Python 3.x
- OpenCV
- dlib
- imutils
- numpy

## Credits
- Developed by Jorge Lopez Casillas and contributors
- Dlib's facial landmark model by Davis King

## License
This project is for educational purposes. See repository for license details.
