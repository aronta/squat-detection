from flask import Response, Flask, jsonify, request, send_from_directory, send_file
from flask_socketio import SocketIO, emit
import argparse
import os
import cv2
import time
from threading import Thread
import numpy as np
from scipy import ndimage
import multiprocessing

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task")

# MediaPipe pose landmark indices
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28

# Body connections used to draw the skeleton overlay
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')
bordersize = 5

@app.route('/')
def main():
    return send_from_directory("page", "main.html")

#BACKUP if socketIO doesn't work, ping server manually from js every 5ms
@app.route('/status_info', methods=['GET'])
def get_status():
    return jsonify("backup")

@app.route('/video_feed')
def video_feed():
    return Response(start_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
  f = open("test.txt", "w")
  f.write("start")
  f.close()
  return "ok", 200

@app.route('/stop', methods=['POST'])
def stop():
  f = open("test.txt", "w")
  f.write("stop")
  f.close()
  return "ok", 200

@app.route('/reset_counter', methods=['POST'])
def reset_counter():
  f = open("test.txt", "w")
  f.write("reset_counter")
  f.close()
  return "ok", 200

def draw_landmarks(image, landmarks):
    h, w = image.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(image, points[a], points[b], (255, 255, 255), 2)
    for p in points:
        cv2.circle(image, p, 4, (0, 0, 255), -1)
    return image

def start_video_stream():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_poses=1)
    detector = PoseLandmarker.create_from_options(options)

    count_of_squats = 0
    squat_pos = 0
    prev_squat_pos = 0
    
    prev_state = "idle"
    videostream = VideoStream().start()

    numb_of_frames = 0
    while True:
        f = open("test.txt", "r")
        tmp = f.read()
        f.close()
        
        if tmp.lower() == "reset_counter":
            count_of_squats = 0
            tmp = prev_state
            f = open("test.txt", "w")
            f.write(prev_state)
            f.close()
        
        if tmp.lower() == "start":
            prev_state = "start"
            image = videostream.read()

            image_h, image_w = image.shape[:2]

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if not result.pose_landmarks:
                is_human = False
            else:
                is_human = True

            try:
                lm = result.pose_landmarks[0]

                def to_px(landmark):
                    return (int(landmark.x * image_w), int(landmark.y * image_h))

                center_11 = to_px(lm[LEFT_HIP])    # left hip
                center_12 = to_px(lm[LEFT_KNEE])   # left knee
                center_13 = to_px(lm[LEFT_ANKLE])  # left ankle

                center_8 = to_px(lm[RIGHT_HIP])    # right hip
                center_9 = to_px(lm[RIGHT_KNEE])   # right knee
                center_10 = to_px(lm[RIGHT_ANKLE]) # right ankle


                squat_left_angle = angle_between_points(center_11, center_12, center_13)
                squat_right_angle = angle_between_points(center_8, center_9, center_10)
                
                squat_pos = 1 if (squat_right_angle <= 120 or squat_left_angle <= 120) else 0
                if prev_squat_pos - squat_pos == 1:
                    socketio.emit("squat_counter", "success")
                    count_of_squats +=1
                    #print("$$$$ LEFT ANGLE (sleep) $$$$: ", squat_left_angle)
                    #print("$$$$ RIGHT ANGLE (sleep) $$$$: ", squat_right_angle)
                    #time.sleep(1)
                socketio.emit("squat_position", "standing")
                prev_squat_pos = squat_pos
                
                if prev_squat_pos == 1:
                    socketio.emit("squat_position", "squat")    
                #print("####################NUMBER OF SQUATS####################", count_of_squats)

            except:
                is_human = False
                
                if squat_pos == 0:
                    socketio.emit("squat_position", "undefined")
                pass                     
            
            if result.pose_landmarks:
                image = draw_landmarks(image, result.pose_landmarks[0])
            image = cv2.resize(image, (0, 0), fx=0.65, fy=0.65)
            
            if is_human or squat_pos:
                image = cv2.copyMakeBorder(image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=(0,255,0))
            else:
                image = cv2.copyMakeBorder(image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=(0,0,255))
            
            ret, jpeg = cv2.imencode('.jpg', image)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            
            numb_of_frames += 1
            
            if cv2.waitKey(1) == 'q':
                break
           
        elif tmp == "stop":
            prev_state = "idle"
            f = open("test.txt", "w")
            f.write("idle")
            f.close()
    videostream.stop()
    detector.close()
    cv2.destroyAllWindows()


class VideoStream:
    def __init__(self):
        # Initialize the PiCamera and the camera image stream
        #breakpoint()
        
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1120)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        socketio.emit("camera_initiated", "true")
        #print("Camera initiated.")
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False
    
    def __del__(self):
        #releasing camera
        self.stream.release()

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True
        
    def get_frame(self):
        
        image = self.read()
        
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def squatAPI():
   f = open("test.txt", "w")
   f.write("start")
   f.close()
   #app.run(host='0.0.0.0', port=7000, debug=True, threaded=True, use_reloader=False)
   socketio.run(app, host='0.0.0.0', port=8000, allow_unsafe_werkzeug=True)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def angle_between_points(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

if __name__ == '__main__':   
    server = multiprocessing.Process(target=squatAPI)
    server.start()
