from flask import Response, Flask, jsonify, request, send_from_directory, send_file
from flask_socketio import SocketIO, emit
import argparse
import cv2
import time
from threading import Thread
import numpy as np
from scipy import ndimage
import multiprocessing

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

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

def start_video_stream():
    w, h = model_wh('96x112')

    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h), trt_bool=str2bool('False'))   

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
            
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=2.5)
                 
            if len(humans) != 1:
                is_human = False
            else:
                is_human = True
                  
            try:
                center_11 = (int(humans[0].body_parts[11].x * w), int(humans[0].body_parts[11].y * h)) # left hip
                center_12 = (int(humans[0].body_parts[12].x * w), int(humans[0].body_parts[12].y * h)) # left knee
                center_13 = (int(humans[0].body_parts[13].x * w), int(humans[0].body_parts[13].y * h)) # left ankle
                
                center_8 = (int(humans[0].body_parts[8].x * w), int(humans[0].body_parts[8].y * h)) # right hip
                center_9 = (int(humans[0].body_parts[9].x * w), int(humans[0].body_parts[9].y * h)) # right knee
                center_10 = (int(humans[0].body_parts[10].x * w), int(humans[0].body_parts[10].y * h)) # right
                
                
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
                #ovaj dio treba testirat
                #smisao ako je ulovio da si uso u cucanj makar ces u sljedecem frameu bit ne human nemoj dat da je krivo
                if squat_pos == 0:
                    socketio.emit("squat_position", "undefined")
                pass                     
            
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
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
   socketio.run(app, host='0.0.0.0')

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
    


