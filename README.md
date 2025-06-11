<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python" />
  <img src="https://img.shields.io/badge/OpenCV-enabled-brightgreen.svg" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Scikit--learn-ML-yellowgreen.svg" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" />
  <img src="https://img.shields.io/badge/PRs-welcome-blue.svg" alt="PRs Welcome" />
</p>

<div align="center">
  <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="Scikit-learn Logo" width="90"/>
</div>

# Monitoring System for Vehicle Drivers Using AI and Machine Learning

A real-time, AI-powered monitoring system that enhances vehicle driver safety by detecting fatigue and drowsiness using advanced computer vision and machine learning techniques.

## 📝 Description

- Developed a real-time fatigue detection system using Python, OpenCV, and Scikit-learn, processing live video feeds to monitor driver alertness in moving vehicles.
- Implemented Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) algorithms using OpenCV, achieving over 85% accuracy in real-time detection of drowsiness and yawning symptoms.
- Designed an early warning system with audio/visual alerts, contributing to reduced fatigue-related incidents during long-distance drives.
- **Tools Used:** Python, OpenCV, Numpy, Scikit-learn, Seaborn

## 🚀 Features

- Real-time video feed monitoring and analysis
- Fatigue and drowsiness detection using EAR & MAR algorithms
- Yawning detection for early symptom identification
- Machine learning-based alertness classification
- Audio and visual alerts to warn drivers in real-time
- Data logging and analytics

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Computer Vision:** OpenCV
- **Machine Learning:** Scikit-learn
- **Numerical Operations:** Numpy
- **Visualization:** Seaborn

## 🧑‍💻 How It Works

1. The system captures live video feed from a camera facing the driver.
2. Facial landmarks are detected, and EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) are continuously calculated.
3. Machine learning models classify driver's alertness based on these features.
4. If signs of fatigue or yawning are detected, the system triggers audio/visual warnings in real-time.

## 📂 Project Structure

```
dms/
├── DMS_Final_Code.py      # Main application code
├── datasheet.txt          # Documentation or data sheet for the project
├── User_Data/             # Stores user data (images, logs, etc.)
├── User_Data_Graph/       # Contains graphs or analytics outputs
├── audio/                 # Audio files for alerts
├── images/                # Image assets for detection or documentation
├── models/                # Machine learning models
├── dist/                  # Distribution or build files (if applicable)
```

## 💻 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sk-2157/dms.git
   cd dms
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is missing, install these packages manually:
   ```bash
   pip install numpy scipy imutils dlib opencv-python Pillow matplotlib pandas
   ```

4. **(Windows only) Install winsound:**
   - winsound is part of the Python standard library on Windows. No action needed on Windows.
   - If you are on Linux or Mac, audio alerts may need to be adapted (e.g., use playsound or another library).

5. **Install system dependencies for dlib and OpenCV:**

   - **Ubuntu/Linux:**
     ```bash
     sudo apt-get update
     sudo apt-get install build-essential cmake
     sudo apt-get install libopenblas-dev liblapack-dev
     sudo apt-get install libx11-dev libgtk-3-dev
     sudo apt-get install python3-dev
     sudo apt-get install libboost-all-dev
     sudo apt-get install libdlib-dev
     sudo apt-get install ffmpeg
     ```

   - **Windows:**
     - Install Visual C++ Build Tools (required for compiling dlib if not using a pre-built wheel).
     - Download and install pre-built binaries for dlib and OpenCV if you encounter build issues.

6. **Make sure the following directories exist (create them if missing):**
   - `User_Data/`
   - `User_Data_Graph/`
   - `audio/`
   - `images/`
   - `models/`
   - `dist/`

7. **Download the facial landmark predictor model:**  
   Download `shape_predictor_68_face_landmarks.dat` from [dlib's model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), extract it, and place it inside the `models/` directory.

8. **Run the application:**
   ```bash
   python DMS_Final_Code.py
   ```

**Notes:**
- If you encounter issues with dlib installation, refer to the [official dlib install instructions](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/).
- The application is designed for Windows due to `winsound` use (audio alerts). For Linux/Mac, consider replacing `winsound.Beep` with a cross-platform solution.


## ▶️ Usage

- Connect a camera to your system.
- Run the main application file:
  ```bash
  python DMS_Final_Code.py
  ```
- Follow on-screen instructions and allow camera access.
- The system will monitor the driver and provide alerts if fatigue or drowsiness symptoms are detected.

## 🤝 Contributing

Contributions and suggestions are welcome! Please open an issue or submit a pull request for improvements.


---

<div align="center">
  <strong>Made with ❤️ using Python, OpenCV, and Scikit-learn</strong>
</div>
