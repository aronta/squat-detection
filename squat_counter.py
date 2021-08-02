from flask import Response, Flask, jsonify, request, send_from_directory, send_file
import argparse
import cv2
import time
from threading import Thread
import numpy as np
from scipy import ndimage
import multiprocessing

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

flag_start = False
app = Flask(__name__)

@app.route('/')
def main():
  return send_from_directory("page", "main.html")

@app.route('/video_feed')
def video_feed():
    print("USO U VIDEO FEEET")
    #videostream_1 = VideoStream().start()
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
  global flag_start
  flag_start = True
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

def gen():
    print("USO U GENNNNNNNNNNNNNNN")
#     camera = VideoStream().start()
#     while True:
#         image = camera.read()
#         print(image)
#         ret, jpeg = cv2.imencode('.jpg', image)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
#         if cv2.waitKey(1) == 'q':
#             break

class VideoStream:
    def __init__(self):
        # Initialize the PiCamera and the camera image stream
        #breakpoint()
        
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1120)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        print("Camera initiated.")
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

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
   f.write("idle")
   f.close()
   app.run(host='0.0.0.0', port=7000, debug=True, threaded=True, use_reloader=False)

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


font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 1.5
fontColor = (255, 255, 255)
lineType = 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')

    parser.add_argument('--leg', type=str, default='right')

    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    w, h = model_wh('96x112')
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(32, 32), trt_bool=str2bool(args.tensorrt))
    
    server = multiprocessing.Process(target=squatAPI)
    server.start()
    
    count_of_squats = 0
    squat_pos = 0
    prev_squat_pos = 0
    
    prev_state = "idle"
    
    #stop_flag = 1
    videostream = VideoStream().start()
    
    i = 0
    while True:
        f = open("test.txt", "r")
        tmp = f.read()
        f.close()
        #print(count_of_squats)
        #print(tmp)
        
        if tmp.lower() == "reset_counter":
            count_of_squats = 0
            tmp = prev_state
            f = open("test.txt", "w")
            f.write(prev_state)
            f.close()
        
        if tmp.lower() == "start":
            prev_state = "start"
#             if stop_flag:
#                 videostream = VideoStream().start()
#                 stop_flag = 0
            image = videostream.read()
            #print(videostream.get_frame())
            
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=2.5)
                 
            if len(humans) != 1:
                continue
            
            if args.leg == 'right':        
                try:
                    center_11 = (int(humans[0].body_parts[11].x * w), int(humans[0].body_parts[11].y * h)) # left hip
                    center_12 = (int(humans[0].body_parts[12].x * w), int(humans[0].body_parts[12].y * h)) # left knee
                    center_13 = (int(humans[0].body_parts[13].x * w), int(humans[0].body_parts[13].y * h)) # left ankle
                    
                    center_8 = (int(humans[0].body_parts[8].x * w), int(humans[0].body_parts[8].y * h)) # right hip
                    center_9 = (int(humans[0].body_parts[9].x * w), int(humans[0].body_parts[9].y * h)) # right knee
                    center_10 = (int(humans[0].body_parts[10].x * w), int(humans[0].body_parts[10].y * h)) # right
                    
                    squat_left_angle = angle_between_points(center_11, center_12, center_13)
                    squat_right_angle = angle_between_points(center_8, center_9, center_10)
                    
                    squat_pos = 1 if (squat_right_angle <= 130 or squat_left_angle <= 130) else 0
                    if prev_squat_pos - squat_pos == 1:
                        count_of_squats +=1
                        print("$$$$ LEFT ANGLE (sleep) $$$$: ", squat_left_angle)
                        print("$$$$ RIGHT ANGLE (sleep) $$$$: ", squat_right_angle)
                        time.sleep(1)
                    prev_squat_pos = squat_pos
                    print("####################NUMBER OF SQUATS####################", count_of_squats)
    #                 cv2.putText(image, 'Number of squats: ' + str(count_of_squats),
    #                     (200, 50),
    #                     font, 
    #                     fontScale,
    #                     fontColor,
    #                     lineType
    #                 )
    #                 cv2.putText(image, 'Angle of knee joint: ' + str(round(squat_right_angle, 1)),
    #                     (200, 150),
    #                     font, 
    #                     fontScale,
    #                     fontColor,
    #                     lineType
    #                 )
    # 
    #                 cv2.putText(image, 'Squat position: ' + str('Yes' if squat_pos==1 else 'No'),
    #                     (200, 250),
    #                     font, 
    #                     fontScale,
    #                     fontColor,
    #                     lineType
    #                 )
    # 
    #                 cv2.putText(image, 'Tracking leg: Right',
    #                     (200, 350),
    #                     font, 
    #                     fontScale,
    #                     fontColor,
    #                     lineType
    #                 )
    #                 cv2.putText(image, 'Burned calories: ' + str(round(52.5 * count_of_squats, 2)),
    #                     (200, 450),
    #                     font, 
    #                     fontScale,
    #                     fontColor,
    #                     lineType
    #                 )

                except:
                    pass                     
            
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            #print("Image shape: ", image.shape)
            i += 1
            #print("Frame:", i)
            if i > 600000:
                cv2.destroyAllWindows()
                videostream.stop()
                
                break
            cv2.imshow('tf-pose-estimation result', cv2.resize(image, (0, 0), fx=0.5, fy=0.5))
            
            if cv2.waitKey(1) == 'q':
                break
           
        elif tmp == "stop":
            prev_state = "idle"
            f = open("test.txt", "w")
            f.write("idle")
            f.close()
            #stop_flag = 1
    videostream.stop()
    cv2.destroyAllWindows()
