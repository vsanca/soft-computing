# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:16:07 2016

@author: student
"""

import cv2
from skimage.io import imshow
from skimage.morphology import remove_small_objects, diamond, disk, rectangle, closing, opening
from skimage.color import rgb2gray
from skimage.measure import label
import numpy as np


def processFrame(frameIn):
    frame_grey = rgb2gray(frameIn)
    frame_bw = frame_grey>0.5
    #frame_bw = closing(frame_bw,diamond(5))
    frame_bw = remove_small_objects(frame_bw,min_size=15)
    return frame_bw


path = "/home/student/Desktop/lvl3/level-3-video-train/videos/video-0.avi"

video = cv2.VideoCapture(path)

#dframe = []

while(1):
    grabbed,frame1 = video.read()
    
    if(grabbed==True):
        frame1 = processFrame(frame1)        
        
        grabbed,frame2 = video.read()
        
        if(grabbed==True):
            frame2 = processFrame(frame2)
            df = abs(frame2-frame1)
            df = opening(df,disk(1)
            #df = closing(df,diamond(5))
            #dframe.append(df)
            frame1 = frame2
            cv2.imshow('img',df*1.0)
            
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break            
            
        else:
            break
    else:
        break


cv2.destroyAllWindows()
video.release()
        