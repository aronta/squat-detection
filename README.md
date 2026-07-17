# Squat-counter
Using pose estimation (tf-pose) to count the number of squats done with Python3 and Flask.<br>Video stream and simple UI can be found on localhost:5000 by running "squat_counter_v2.py".<br>
Made to be used on <b>Raspberry Pi 4</b> with supported camera module.


## Credits for the model implementation [this repo](https://github.com/ildoonet/tf-pose-estimation)

- Implemented the use of posenet model and ran it locally setting up the environment.

## Run on Raspberry Pi (original tf-pose version)
`squat_counter_v2.py` is the original script, built for a <b>Raspberry Pi 4</b> with a supported camera module (Pi Camera or a USB webcam). It uses [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation) (backed by TensorFlow 1.x) for pose estimation and serves a Flask + Socket.IO web UI on port `5000`.

> Note: `tf-pose` / TensorFlow 1.x only work on older Python (roughly 3.5–3.7). On modern desktops (especially Apple Silicon macOS) this stack will not install — use the [MediaPipe version](#run-on-macos-mediapipe-version) below instead.

### Prerequisites
- Raspberry Pi 4 (64-bit Raspberry Pi OS recommended)
- Pi Camera module (enabled via `sudo raspi-config` → Interface Options → Camera) or a USB webcam
- Python 3.7 with `pip` and `venv`
- System libraries for OpenCV:
```bash
sudo apt update
sudo apt install -y python3-venv python3-dev libatlas-base-dev libjpeg-dev \
                    libhdf5-dev libqtgui4 libqt4-test
```

### Setup
```bash
# From the repo root
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Core dependencies
pip install -r requirements.txt        # opencv-contrib-python, numpy, tf-pose
pip install flask flask-socketio scipy

# tf-pose needs TensorFlow 1.x (install the build that matches your Pi/Python)
pip install "tensorflow==1.15.*"
```

`tf-pose` also builds a C extension (`pafprocess`) with SWIG. If installation fails, install SWIG and build tools first:
```bash
sudo apt install -y swig build-essential
```
See the [tf-pose-estimation repo](https://github.com/ildoonet/tf-pose-estimation) for full model/setup details.

### Run
```bash
➜ python squat_counter_v2.py
```
Then open <b>http://localhost:5000</b> in your browser (or `http://<raspberry-pi-ip>:5000` from another device on the same network — the server binds to `0.0.0.0`).

### Usage
1. Wait for the loading spinner to disappear (the camera has initialised).
2. Click <b>Stop</b> then <b>Start</b> (the Start button begins disabled in the UI).
3. Step back until the border around the video turns <b>green</b> (camera can see you head to toe); while it is <b>red</b>, reposition inside the camera's view.
4. Squat — the counter increments on each rep. Use <b>Reset Counter</b> to start over.

### How it works
- The camera is read in a background thread (`VideoStream`) and each frame is run through the tf-pose `mobilenet_thin` model.
- The angle at each knee (hip → knee → ankle) is computed; a knee angle ≤ 120° counts as the "down" position, and returning to standing increments the squat counter.
- Detection state (`start` / `stop` / `reset_counter` / `idle`) is passed between the Flask routes and the video loop via the `test.txt` file, and live updates are pushed to the browser over Socket.IO.

### tf-pose body part indices
The tf-pose model outputs these body parts (the squat logic uses the hip/knee/ankle indices):
```
0: Nose
1: Sternum
2: Right Shoulder
3: Right Elbow
4: Right Palm
5: Left Shoulder
6: Left Elbow
7: Left Palm
8: Right Hip
9: Right Knee
10: Right Ankle
11: Left Hip
12: Left Knee
13: Left Ankle
14: Right Eye
15: Left Eye
16: Right Ear
17: Left Ear
```

## Run on macOS (MediaPipe version)
`tf-pose` depends on TensorFlow 1.x, which has no builds for modern Python or Apple Silicon, so it cannot run on macOS. `squat_counter_mediapipe.py` is a drop-in replacement that swaps `tf-pose` for [MediaPipe](https://github.com/google-ai-edge/mediapipe) (runs natively on Apple Silicon via the GPU/Metal). It keeps the same Flask + Socket.IO server, web UI, and squat-angle logic.

### Setup
```bash
# From the repo root
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-mediapipe.txt
```

The MediaPipe pose model (`pose_landmarker.task`) is already included in the repo. If it is missing, download it with:
```bash
curl -sSL -o pose_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
```

### Run
```bash
➜ python squat_counter_mediapipe.py
```
Then open <b>http://localhost:8000</b> in your browser.

> Port `8000` is used instead of `5000` because macOS reserves port `5000` for the AirPlay Receiver.

### Usage
1. On first launch, macOS will ask to grant <b>camera permission</b> to your terminal/Python — allow it.
2. In the UI, click <b>Stop</b> then <b>Start</b> (the Start button begins disabled).
3. Step back until the border around the video turns <b>green</b> (camera can see you head to toe).
4. Squat — the counter increments on each rep. Use <b>Reset Counter</b> to start over.

### MediaPipe body part indices (used by `squat_counter_mediapipe.py`)
```
23: Left Hip     24: Right Hip
25: Left Knee    26: Right Knee
27: Left Ankle   28: Right Ankle
```
