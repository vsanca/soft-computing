# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 21:28:02 2016

@author: Viktor
"""

import numpy as np

"""Moves 28x28 ``image`` into upper left corner from original position defined by ``x`` and ``y``"""
def move(image,x,y):
    img = np.zeros((28,28))
    img[:(28-x),:(28-y)] = image[x:,y:]

    return img
    
"""Fills ``image`` with black pixels to the size of 28x28"""
def fill(image):
    if(np.shape(image)!=(28,28)):
        img = np.zeros((28,28))
        x = 28 - np.shape(image)[0]
        y = 28 - np.shape(image)[1]
        img[:-x,:-y] = image
        return img
    else:
        return image

"""Check whether regions overlap. ``c1``, ``c2`` are the centroids of regions, while ``par`` is given distance."""
def check(c1,c2,par=15):
    if(abs(c1[0]-c2[0])<=par and abs(c1[1]-c2[1])<=par):
        return True
    else:
        return False

"""Merge two regions from ``image`` defined by ``bbox1`` and ``bbox2``."""
def merge(image,bbox1,bbox2):
    if(bbox1[0]<bbox2[0]):
        x=bbox1[0]
    else:
        x=bbox2[0]

    if(bbox1[1]<bbox2[1]):
        y=bbox1[1]
    else:
        y=bbox2[1]

    return(image[x:x+28,y:y+28])
   
"""Helper function."""
def check2(indices,i):
    check = False
    for el in indices:
        if(el==i):
            check = True
            break
        
    return check

"""Loads and returns a list of image paths from input file given in ``path``."""
def ucitavanje(path):
    image_path = []

    with open(path) as f:
        data = f.read()
        lines = data.split('\n')
        for i, line in enumerate(lines):
            if(i>1):
                cols = line.split('\t')
                if(cols[0]!=''):
                    image_path.append(cols[0])
                
        f.close()
        
    return image_path

"""Converts RGB image (``img_rgb``) to Grayscale, and returns converted image."""
def my_rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    img_gray = 0.5*img_rgb[:, :, 0] + 0.0*img_rgb[:, :, 1] + 0.5*img_rgb[:, :, 2]
    img_gray = img_gray.astype('uint8')
    return img_gray
