# commaai_calib_code

My solution to comma ai's calibration challenge: [https://github.com/commaai/calib_challenge?tab=readme-ov-file]
MSE of <25% on the test data

# Video Demo 
[https://youtube.com/shorts/EXIoAvla1Tc?feature=share]

---

# KalmanFilter Lane Detection and Calibration Challenge Solution

This repository contains an implementation of lane detection using Kalman Filters for detecting and tracking curved lane lines in video sequences. The solution is tailored for the comma.ai calibration challenge and integrates techniques for estimating pitch and yaw angles from lane geometry, enforcing parallelism, and estimating vehicle speed.

---

## Features

- **Kalman Filter Integration**: Smooths and predicts lane lines over consecutive frames.
- **Curved Lane Detection**: Identifies and filters curved lane lines based on curvature and slope.
- **Pitch and Yaw Calculation**: Estimates pitch and yaw angles using vanishing points and lane line geometry.
- **Speed Estimation**: Uses optical flow between frames to estimate the vehicle's speed.
- **Parallelism Enforcement**: Ensures left and right lane lines are parallel for consistency.
- **Lane Interpolation**: Fills gaps in lane detections for smoother tracking.

---

## Prerequisites

- **Python** 3.7 or higher
- **OpenCV** 4.x
- **NumPy** 1.21 or higher

Install dependencies using:
```bash
pip install numpy opencv-python

