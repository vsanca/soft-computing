# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:16:07 2016

@author: student
"""

import cv2
import numpy as np

class Tracker:
    centroid = (0,0)
    bbox = []
    rect = ((0, 0),(0, 0), 0)
    code = 0
    intersected = False
    
    def __init__(self, centroid, bbox, rect, code):
        self.centroid = centroid
        self.bbox = bbox
        self.rect = rect
        self.code = code
        
    def update(self, centroid, bbox, rect):
        self.centroid = centroid
        self.bbox = bbox
        self.rect = rect


def processFrame(frameIn):
    frame_no_green = frameIn
    frame_grey = cv2.cvtColor(frame_no_green,cv2.COLOR_BGR2GRAY)
    ret, frame_bw = cv2.threshold(frame_grey,170,255,0)
    
    frame_bw = cv2.morphologyEx(frame_bw, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    
    return frame_bw

    
def getContours(frame_bw):
    contours = cv2.findContours(frame_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    return contours

    
def getBoundingRectsAndCentroids(contours,thresh=0):
    
    bboxes = []
    centroids = []
    rects = []
    
    for c in contours:
        if(cv2.contourArea(c)>5):
            rect = cv2.minAreaRect(c)
            rects.append(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            bboxes.append(box)
            #centroid calculation
            M = cv2.moments(c)
            cx = (M['m10']/M['m00'])
            cy = (M['m01']/M['m00'])
            centroids.append((cx,cy))
            
    return bboxes,centroids,rects
 
    
def drawBoundingRects(frame,rects):
    for box in rects:
        cv2.drawContours(frame,[box],0,(0,180,180),2)

def drawCentroids(frame,centroids):
    for c in centroids:
        cx = int(c[0])
        cy = int(c[1])
        cv2.circle(frame,(cx,cy),1,(0,0,255), -1)
        
def drawCode(frame,trackers):
    for tracker in trackers:
        c = tracker.centroid
        cx = int(c[0])
        cy = int(c[1]) 
        cv2.putText(frame,str(tracker.code),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0))
    
def drawAll(frame,rects,centroids):
    drawBoundingRects(frame,rects)
    drawCentroids(frame,centroids)
    

def checkNeighbour(centroid1,centroid2,tolerance):
    if((abs(centroid1[0]-centroid2[0])<=tolerance) and (abs(centroid1[1]-centroid2[1])<=tolerance)):
        return True
    else:
        return False
        
        
        
def checkForRemoval(centroid,x,y,tolerance):
    if(x-centroid[0]<=tolerance or y-centroid[1]<=tolerance):
        return True
    else:
        return False
            
        
def addToTrackerList(tracked_list, tracker):
    
    tmp = []#tracked_list
    
    check = False
    toAddNew = True

    for t in tracked_list:
        if(t.intersected == False):
            threshold = 15
        else:
            threshold = 30
            
        check = checkNeighbour(t.centroid,tracker.centroid,threshold)
        if(checkForRemoval(t.centroid,640,480,20)==False):
            if(check==True):
                #if(t.intersected == False):
                tracker.code = t.code #preuzmi identifikaciju
                tracker.intersected = t.intersected
                tmp.append(tracker)
                
                toAddNew = False
            else:
                tmp.append(t)
    
    global counter
    
    #Ako se do kraja ni sa kim nije poklopio, treba ga dodati kao novi objekat
    if(toAddNew):
        tracker.code = counter #dodeli novu identifikaciju
        counter = counter+1
        tmp.append(tracker)
            
    return tmp
    

def createTrackersFromBoxesAndCentroids(bboxes,centroids,rects):
    trackers = []

    for i in range(len(bboxes)):
        tmp = Tracker(centroids[i],bboxes[i],rects[i],-1)
        trackers.append(tmp)
        
    return trackers
        
  
counter = 0
      
path = "E:\\FTN\\7. semestar\\SC\\Lvl3\\level-3-video-train\\level-3-video-train\\videos\\video-7.avi"
tracked_objects = []

video = cv2.VideoCapture(path)

video.isOpened()

total = 0
id_touched = []

while(1):
    grabbed,frame1 = video.read()
    
    if(grabbed==True):
        
        #linija
        empty_frame = np.ones_like(frame1)
        cv2.line(empty_frame,(100,450),(500,100),(200,200,200), 2)
        line_contour = getContours(processFrame(empty_frame))
        line_bbox,line_centroid,line_rect = getBoundingRectsAndCentroids(line_contour)
        
        drawBoundingRects(empty_frame,line_bbox)
        
        
        #ostatak
        frame = frame1.copy()   #za prikaz
        frame1 = processFrame(frame1)     
                
        contours = getContours(frame1)
        
        bboxes,centroids,rects = getBoundingRectsAndCentroids(contours)
        
        trackers = createTrackersFromBoxesAndCentroids(bboxes,centroids,rects)
        
        for t in trackers:
            tracked_objects = addToTrackerList(tracked_objects, t)
            
        for t in tracked_objects:
            res = cv2.rotatedRectangleIntersection(line_rect[0],t.rect)
            if(res[1]!=None and res[0]==1):
                already_touched = False
                for el in id_touched:
                    if(el == t.code):
                        already_touched = True
                        break
                if(already_touched==False):
                    id_touched.append(t.code)
                    if(t.intersected == False):
                        total = total+1
                    else:
                        t.intersected = True
            
        #print("no of tracked objects: "+str(len(tracked_objects)))
        #print("no of centroids: "+str(len(centroids)))
            
        #empty_frame for model only
        drawAll(frame,bboxes,centroids)
        #cv2.line(frame,(100,450),(500,100),(200,200,200), 2)
        drawCode(frame,tracked_objects)
        cv2.putText(frame,"Total: "+str(total),(450,80),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,0))
        
        cv2.imshow('img',frame)
        
        
        k = cv2.waitKey(15) & 0xff
        if k == 15:
            break            

    else:
        break


cv2.destroyAllWindows()
video.release()

print("TOTAL: "+str(total))