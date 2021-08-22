#Import necessary libraries
import imutils
from flask import Flask, render_template, Response, request, redirect
from dlib_sleep_detector import SLEEP_THRESHOLD_SECS, detect_draw_eyes, set_uploaded_alarm
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils import face_utils
import cv2
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import os

DETECTION_ON = True

#Initialize the Flask app
app = Flask(__name__)

camera = cv2.VideoCapture(0)
time.sleep(1)

# use a method to grab the root directory of the current device
app.config["SOUND_UPLOADS"] = "C:\\Users\\Hannah\\Yawnly-DrowsinessDetector\\assets"
app.config["ALLOWED_SOUND_EXTENSIONS"] = ["WAV", "MP3"]


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

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print("hello")
        print(request.form["mins"])
        print(request.form["secs"])

        if request.files:
            sound = request.files["sound"]

            if sound.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowed_sound(sound.filename):
                filename = secure_filename(sound.filename)
                sound.save(os.path.join(
                    app.config["SOUND_UPLOADS"], filename))
                print("sound saved")
                return redirect(request.url)

            else:
                print("That file extension is not allowed")
                return redirect(request.url)

    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def allowed_sound(filename):
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_SOUND_EXTENSIONS"]:
        return True
    else:
        return False

if __name__ == "__main__":
    app.run(debug=True)
