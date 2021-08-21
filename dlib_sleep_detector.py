# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from flask import Response
from flask import Flask
from flask import render_template
import threading
import numpy as np
#import argparse
import imutils
import time
import dlib
import cv2
from datetime import datetime
#from playsound import playsound

EYES_CLOSED_FLAG = 0	#1 if eyes closed, 0 otherwise
sleep_start = None 	# time first asleep; None = never asleep
CLOSED_EAR_THRESHOLD = 10 	#All EAR under this threshold is considered closed eyes
SLEEP_THRESHOLD_SECS = 5	#Number of seconds until considered asleep
MIN_SLEEP_TIME = 5		# Minimum number of seconds of eyes closed to be considered 'asleep'

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
#facerec = dlib.face_recognition_model_v1("./dlib_face_recognition_resnet_model_v1.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def detect_draw_eyes (frame, grayframe):
	global EYES_CLOSED_FLAG, sleep_start

	timediff = 0
	rects = detector(grayframe, 0)
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(grayframe, rect)
		shape = face_utils.shape_to_np(shape)
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		#print(ear)

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

	if len(rects) > 0:
		if EYES_CLOSED_FLAG == 0 and ear <= CLOSED_EAR_THRESHOLD:
			EYES_CLOSED_FLAG = 1
			sleep_start = datetime.now()		# time object recording time when eyes first closed
			print("sleep started")	
		
		#else if EYES_CLOSED_FLAG == 1 and ear <= CLOSED_EAR_THRESHOLD:
		elif EYES_CLOSED_FLAG == 1 and ear > CLOSED_EAR_THRESHOLD:
			EYES_CLOSED_FLAG = 0
			sleep_end = datetime.now()

			if sleep_start != None:	# sleep has started
				timediff = sleep_end-sleep_start
				if timediff.seconds >= MIN_SLEEP_TIME:
					print("You were asleep for " + str(timediff) + " seconds")
					#print("You slept from "+datetime.strftime(sleep_start,'%m/%d/%Y') +" to "+datetime.strftime(sleep_end,'%m/%d/%Y’))
				sleep_start = None	
	return timediff, frame

		# show the frame
	#cv2.imshow("Frame", frame)


	
	#key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	#if key == ord("q"):
	#	break
# do a bit of cleanup
#cv2.destroyAllWindows()
#vs.stop()
