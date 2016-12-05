# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:27:22 2016

@author: Viktor
"""

import numpy as np
from sklearn.datasets import fetch_mldata
from skimage.io import imread
from sklearn.neighbors import KNeighborsClassifier

#ucitavanje MNIST dataseta
mnist = fetch_mldata('MNIST original')


def test():
    res = {}
    n = 0
    with open('level-1-mnist-train/level-1-mnist/res.txt') as file:	
        data = file.read()
        lines = data.split('\n')
        for id, line in enumerate(lines):
            if(id>0):
                cols = line.split('\t')
                if(cols[0] == ''):
                    continue
                cols[1] = cols[1].replace('\r', '')
                res[cols[0]] = cols
                n += 1
    
    correct = 0
    student = []
    with open("level-1-mnist-train/level-1-mnist/out.txt") as file:	
        data = file.read()
        lines = data.split('\n')
        for id, line in enumerate(lines):
            cols = line.split('\t')
            if(cols[0] == ''):
                continue
            if(id==0):
                student = cols  
            elif(id>1):
                cols[1] = cols[1].replace('\r', '')
                if (res[cols[0]] == cols):
                    correct += 1
                else:
                    print('bad: '+cols[0])
    
    print student
    print 'Tacnih:\t'+str(correct)
    print 'Ukupno:\t'+str(n)
    print 'Uspeh:\t'+str(100*correct/n)+'%'

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

def upis(path,image_path,result):
    with open(path,'w') as f:
        f.write('RA 1/2013 Viktor Sanca\n')
        f.write('file\tsum\n')
        for i in range(0,len(image_path)):
            f.write(image_path[i]+'\t'+str(result[i])+'\n')
        
        f.close()
    
def get_img(image_path):
    img = imread(image_path)
    gray = rgb2gray(img)
    return gray
    
def rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    img_gray = 0.8*img_rgb[:, :, 0] + 0.0*img_rgb[:, :, 1] + 1*img_rgb[:, :, 2]
    img_gray = img_gray.astype('uint8')
    return img_gray

def mark_indices(image):
    starting_indices = []
    img = image.reshape(640*480)

    for i in range(0,(640)*(480-28)):
        if(img[i]<40 and img[i+27]<40 and img[i+27*(640)]<40 and img[i+27*(640)+27]<40):
            starting_indices.append(i)
            
    return starting_indices

def get_image_from_indice(image,start_indice):
    image28_28 = np.empty((28*28),dtype='uint8')
    img = image.reshape(640*480)
    
    for i in range(0,28):
        for j in range(0,28):
            image28_28[28*i+j]=img[start_indice+i*(640)+j]

    return image28_28
    
 

image_path = []
result = []
    
train_out = 'level-1-mnist-train/level-1-mnist/out.txt'
test_out = 'level-1-mnist-test/level-1-mnist-test/out.txt'

train_path = 'level-1-mnist-train/level-1-mnist/'
test_path = 'level-1-mnist-test/level-1-mnist-test/'

image_paths = ucitavanje(test_out)

knn = KNeighborsClassifier(n_neighbors=1,weights='uniform',algorithm='auto', n_jobs=-1)
knn.fit(mnist.data,mnist.target)

suma = [0.0]*(len(image_paths))

for i in range(0,len(image_paths)):
    print('Image: '+str(i+1)+'/'+str(len(image_paths)))
    #podesiti odgovarajuci path
    img = get_img(test_path+image_paths[i])

    start_indices = mark_indices(img.reshape(640*480))
    
    for start_indice in start_indices:
        img_d = get_image_from_indice(img,start_indice)
        nr = knn.predict(img_d.reshape(1,-1))
        suma[i] = suma[i] + nr[0]
        
upis(test_out, image_paths, suma)