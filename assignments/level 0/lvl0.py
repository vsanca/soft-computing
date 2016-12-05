# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:04:05 2016

@author: Viktor
"""

import numpy as np
from skimage.io import imread
from skimage.io import imshow
from skimage.color import rgb2gray
from skimage.filters.rank import otsu
from skimage.morphology import opening, closing
import matplotlib.pyplot as plt
from scipy import ndimage

def my_rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))  # zauzimanje memorije za sliku (nema trece dimenzije)
    img_gray = 0*img_rgb[:, :, 0] + 0*img_rgb[:, :, 1] + 1*img_rgb[:, :, 2]
    img_gray = img_gray.astype('uint8')  # u prethodnom koraku smo mnozili sa float, pa sada moramo da vratimo u [0,255] opseg
    return img_gray


def processing(path):
    img = imread(path)
    gray = my_rgb2gray(img)

    binary = 1 - (gray > 0.5)
    binary = closing(binary)
    binary = opening(binary)

    labeled, nr_objects = ndimage.label(binary)
    return nr_objects




image_path = []
result = []

with open('Test/out.txt') as f:
    data = f.read()
    lines = data.split('\n')
    for i, line in enumerate(lines):
        if(i>1):
            cols = line.split('\t')
            image_path.append(cols[0])
            
    f.close()
        
for imgs in image_path:
    if(imgs!=''):
        result.append(processing('Test/'+imgs))

with open('Test/out.txt','w') as f:
    f.write('RA 1/2013 Viktor Sanca\n')
    f.write('file\tcircles\n')
    for i in range(0,len(image_path)-1):
        f.write(image_path[i]+'\t'+str(result[i])+'\n')
    
    f.close()


incorrect = ['images/img-13.png',
 'images/img-17.png',
 'images/img-20.png',
 'images/img-35.png',
 'images/img-42.png',
 'images/img-45.png',
 'images/img-51.png',
 'images/img-58.png',
 'images/img-87.png',
 'images/img-96.png']

path = 'Trening/'+incorrect[0]

img = imread(path)
imshow(img)

gray = my_rgb2gray(img)   
imshow(gray)    
    
binary = 1 - (gray > 0.5)
binary = closing(binary)
binary = opening(binary)
    
imshow(binary)        
    
labeled, nr_objects = ndimage.label(binary)
print nr_objects