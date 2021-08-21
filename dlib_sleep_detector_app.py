# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
#import argparse
import imutils
import time
import dlib
import cv2
from datetime import datetime

CLOSED_EAR_THRESHOLD = 0.2 	#All EAR under this threshold is considered closed eyes
SLEEP_THRESHOLD_SECS = 5	#Number of seconds until considered asleep
EYES_CLOSED_FLAG = 0	#1 if eyes closed, 0 otherwise

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio (EAR)
	#ear = (A + B) / (2.0 * C)
	ear = A + B
	# return the eye aspect ratio
	return ear

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
#fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(2.0)

# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	#if fileStream and not vs.more():
	#	break
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	#print(rects)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		print(ear)

				# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

	if EYES_CLOSED_FLAG == 0 and ear <= CLOSED_EAR_THRESHOLD:
		EYES_CLOSED_FLAG = 1
		sleep_start = datetime.now()		# time object recording time when eyes first closed	

	if EYES_CLOSED_FLAG == 1 and ear > CLOSED_EAR_THRESHOLD:
		EYES_CLOSED_FLAG = 0
		sleep_end = datetime.now()
		timediff = sleep_end-sleep_start
		if timediff.seconds >= SLEEP_THRESHOLD_SECS:
			print("You were asleep for " + str(timediff) + " seconds")
			#print("You slept from "+datetime.strftime(sleep_start,'%m/%d/%Y') +" to "+datetime.strftime(sleep_end,'%m/%d/%Y’))

		# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()