# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 14:38:37 2016

@author: Viktor
"""



import cv2

video_path = 'E:/FTN/7. semestar/SC/Lvl3/level-3-video-train/level-3-video-train/videos/video-0.avi'

video = cv2.VideoCapture(video_path)

video.isOpened()

first_frame = None


(grabbed,frame)=video.read()

while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = video.read()
	text = "Unoccupied"
 
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break
 
	# resize the frame, convert it to grayscale, and blur it
	#frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# if the first frame is None, initialize it
	if first_frame is None:
		firstFrame = gray
		continue