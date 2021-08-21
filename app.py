#Import necessary libraries
import imutils
from flask import Flask, render_template, Response
from dlib_sleep_detector import detect_draw_eyes
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils import face_utils
import cv2

from datetime import datetime
#Initialize the Flask app
app = Flask(__name__)

camera = cv2.VideoCapture(0)

def generate():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timediff, frame = detect_draw_eyes (frame, gray)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
