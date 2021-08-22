# Yawnly-DrowsinessDetector

# What it does
Yawnly is a web application that encourages students and workers to stay awake by sounding an alarm whenever it detects the user’s eyes to be closed. The user can set how long the eyes can remain closed before the alarm goes off. Also, the user can set the alarm with a personalized sound according to their needs. While your online meeting is running, just turn off your camera and microphone and keep Yawnly open in another tab to wake you up if you happen to doze off!

# How we built it
To build Yawnly, we used HTML CSS and Javascript for our front-end. For the back-end, we wrote our code in Python and used the Flask library to integrate the backend with the frontend of our web application. For the sleep detector, we used OpenCV to capture and stream the video and facial landmarks in the dlib C++ library, a pre-trained machine learning face detector, to detect the eyes from frames of the video stream. The facial landmarks provide coordinates for points at the top and bottom of both eyes. We use a custom algorithm adding the Euclidean distance between the top and bottom coordinates of each eye that classifies an eye as closed if the average output of both eyes is below a predetermined threshold.
