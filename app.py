#Import necessary libraries
import imutils
from flask import Flask, render_template, Response, request
from dlib_sleep_detector import SLEEP_THRESHOLD_SECS, detect_draw_eyes, set_uploaded_alarm
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils import face_utils
import cv2
import time
from datetime import datetime

DETECTION_ON = True

#Initialize the Flask app
app = Flask(__name__)

camera = cv2.VideoCapture(0)
time.sleep(1)
def generate():
    global DETECTION_ON
    frame_count = 0  
    while True:
        #time.sleep(0.5)
        success, frame = camera.read()  # read the camera frame
        if not success:
            camera.release()
            break
        else:
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if DETECTION_ON:
                if frame_count % 3 == 0:
                    timediff, frame = detect_draw_eyes (frame, gray)
                    if timediff.total_seconds() >= SLEEP_THRESHOLD_SECS:
                        DETECTION_ON = False
                        #camera.release()
                        #break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        frame_count += 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
