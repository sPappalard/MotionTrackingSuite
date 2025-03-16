# Motion Tracking Suite

A Python application that uses computer vision to perform real-time motion tracking with different modes: hand tracking, face mesh detection, pose estimation, and holistic tracking.

## Features

- **Multiple Tracking Modes:**
  - Hand Tracking - Detect and track hand landmarks and gestures
  - Face Mesh - Detailed facial landmark tracking with 468 points
  - Pose Estimation - Full body posture tracking
  - Holistic Tracking - Combined tracking of face, pose, and hands

- **User-Friendly Interface:**
  - Modern, clean UI built with Tkinter
  - Intuitive card-based menu system
  - Real-time FPS counter
  - Loading animations between screens

- **Robust Error Handling:**
  - Graceful recovery from camera errors
  - Dependency verification on startup
  - Comprehensive error messaging

## Screenshots

![Main Menu](/TrackingRealTimeSystem/Media/1.png)
*Main menu with tracking mode selection cards*

![Hand Tracking](/TrackingRealTimeSystem/Media/2.png)
*Hand tracking mode with landmark visualization*

![Face Mesh](/TrackingRealTimeSystem/Media/3.png)
*Face mesh tracking with detailed facial landmarks*

![Pose Estimation](/TrackingRealTimeSystem/Media/4.png)
*Full body pose estimation tracking*

![Olistic Tracking](/TrackingRealTimeSystem/Media/5.png)
*Combined tracking of face, pose, and hands*

## Video Demo

[Watch the demo video](/TrackingRealTimeSystem/Media/6.gif)

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe
- NumPy
- PIL (Pillow)
- Tkinter (usually comes with Python)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/motion-tracking-suite.git
   cd motion-tracking-suite
   ```

2. Install the required dependencies:
   ```
   pip install opencv-python mediapipe numpy pillow
   ```

3. Run the application:
   ```
   python tracking_app.py
   ```

## Code Structure

### Main Classes

#### `TrackingApp`
The core class that manages the entire application. It:
- Initializes the UI and MediaPipe models
- Creates the main menu interface
- Handles starting and stopping tracking
- Processes video frames with the selected tracking mode
- Manages error handling and resource cleanup

#### `ModernTheme`
Defines the visual styling for the application:
- Color scheme constants
- Style configurations for buttons, frames, and labels
- Sets up a consistent visual language throughout the app

#### `LoadingAnimation`
Manages the animated loading indicator:
- Creates an animated circular loading indicator
- Controls animation states (start/stop)
- Renders frames with varying opacity for a smooth effect

### Key Functions

#### `process_frame()`
The heart of the tracking functionality. For each video frame:
- Converts the frame to RGB (required by MediaPipe)
- Processes the frame with the selected MediaPipe model
- Draws landmarks and connections on the frame
- Adds information overlay with FPS counter and mode indicator
- Handles exceptions with error counting and recovery

#### `start_tracking()`
Initializes the tracking mode:
- Sets up the UI for video display
- Opens the webcam
- Creates the video canvas
- Starts the frame update loop

#### `update_frame()`
Continuous frame processing loop:
- Captures frames from the webcam
- Processes frames with the selected tracking mode
- Resizes and displays the processed frames
- Schedules the next frame update
- Handles errors with appropriate recovery

## How It Works

1. **Initialization:**
   - MediaPipe models for hand, face, pose, and holistic tracking are initialized
   - The modern UI theme is applied
   - The main menu is created with selectable tracking modes

2. **Tracking Mode Selection:**
   - User selects a tracking mode from the main menu
   - A loading animation is displayed during initialization
   - The selected model is activated, and webcam capture begins

3. **Real-time Processing:**
   - Video frames are continuously captured from the webcam
   - Each frame is processed by the selected MediaPipe model
   - Visual landmarks are drawn on the frames
   - Processed frames are displayed in real-time
   - FPS and mode information are overlaid on the video

4. **Error Handling:**
   - The application monitors for errors during frame processing
   - If too many consecutive errors occur, tracking is stopped
   - Resources are properly released when the application closes

## MediaPipe Models

The application uses several MediaPipe models:

- **Hands** - Detects 21 landmarks on each hand
- **Face Mesh** - Tracks 468 facial landmarks
- **Pose** - Tracks 33 body landmarks
- **Holistic** - Combines face, pose, and hand tracking into one model

## License

This project is licensed under the MIT License --->  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google for the ML models
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [Tkinter](https://docs.python.org/3/library/tkinter.html) for the GUI framework
