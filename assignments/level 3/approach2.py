# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 21:44:32 2016

@author: student
"""

import cv2
from skimage.io import imshow, show, imshow_collection
from skimage.morphology import remove_small_objects, diamond, disk, rectangle, closing, watershed
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def processFrame(frameIn):
    frame_grey = rgb2gray(frame)
    frame_bw = frame_grey>0.5
    frame_closing = closing(frame_bw,disk(10))
    frame_processed = remove_small_objects(frame_closing,min_size=32)
    return frame_processed

def getCentroids(frame):
    labels = label(frame)
    regions = regionprops(labels)
    
    centroids = []
    for el in regions:
        centroids.append(el.centroid)
    
    return centroids
    
def centroidsAndProcess(frame):
    return getCentroids(processFrame(frame))
    
def centroids2XY(centroids):
    y, x = zip(*centroids)
    
    return list(x), list(y)

def plotCentroids(all_centroids):
    x = []
    y = []    
    
    for el in all_centroids:
        tmp_x, tmp_y = centroids2XY(el)
        x.extend(tmp_x)
        y.extend(tmp_y)
    
    plt.gca().invert_yaxis()
    plt.scatter(x,y)



path = "/home/student/Desktop/lvl3/level-3-video-train/videos/video-0.avi"

video = cv2.VideoCapture(path)


all_centroids = []

for i in range(500):
    (grabbed,frame)=video.read()
    #all_centroids.append(centroidsAndProcess(frame))
    f = processFrame(frame)
    if(i%10==0):
        print(str(i)+"/500")

plotCentroids(all_centroids)