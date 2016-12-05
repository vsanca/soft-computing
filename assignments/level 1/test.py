# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:27:22 2016

@author: Viktor
"""

import numpy as np
from sklearn.datasets import fetch_mldata

from skimage.io import imread
from skimage.morphology import opening, closing
from scipy import ndimage
from skimage.io import imshow


from sklearn.neighbors import KNeighborsClassifier


#ucitavanje MNIST dataseta
mnist = fetch_mldata('MNIST original')


#iscitavanje dataseta i smestanje u matricu radi lakseg pristupa
numbers = [0]*10

numbers[0] = mnist['data'][np.where(mnist['target'] == 0.)[0]]
numbers[1] = mnist['data'][np.where(mnist['target'] == 1.)[0]]
numbers[2] = mnist['data'][np.where(mnist['target'] == 2.)[0]]
numbers[3] = mnist['data'][np.where(mnist['target'] == 3.)[0]]
numbers[4] = mnist['data'][np.where(mnist['target'] == 4.)[0]]
numbers[5] = mnist['data'][np.where(mnist['target'] == 5.)[0]]
numbers[6] = mnist['data'][np.where(mnist['target'] == 6.)[0]]
numbers[7] = mnist['data'][np.where(mnist['target'] == 7.)[0]]
numbers[8] = mnist['data'][np.where(mnist['target'] == 8.)[0]]
numbers[9] = mnist['data'][np.where(mnist['target'] == 9.)[0]]


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
    #gray = closing(gray)
    #gray = opening(gray)
    #binary = (gray < 0.5)
    return gray

def binarize(img):
    return img>1
    
def rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    img_gray = 0.8*img_rgb[:, :, 0] + 0.2*img_rgb[:, :, 1] + 1*img_rgb[:, :, 2]
    img_gray = img_gray.astype('uint8')
    return img_gray

def mark_indices(image):
    starting_indices = []
    img = image.reshape(640*480)

    for i in range(0,(640)*(480-28)):
        if(img[i]<10 and img[i+27]<10 and img[i+27*(640)]<10 and img[i+27*(640)+27]<10):
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

image_paths = ucitavanje(train_out)

#knn = KNeighborsClassifier(n_neighbors=100)
knn = KNeighborsClassifier(n_neighbors=2000,weights='distance',algorithm='auto',n_jobs=-1)
knn.fit(mnist.data,mnist.target)

suma = [0.0]*len(image_paths)

img = get_img(train_path+image_paths[0])

start_indices = mark_indices(img.reshape(640*480))
    
img_d = get_image_from_indice(img,start_indices[10])
nr = knn.predict(img_d)

print(nr)
imshow(img_d.reshape(28,28))

suma[i] = suma[i] + nr[0]
suma[i] = int(suma[i])
        

pogresni = [[],[],[],[],[],[],[],[],[],[]]

for j in range(0,10):
    for i in range(0,1000):
        if(i%100==0):
            print str(i)+'/'+str(1000)
        if(knn.predict(numbers[j][i].reshape(1,-1))!=j):
            pogresni[j].append(i)

print(pogresni)


for i in range(0,10):
    for j in range(0,len(numbers[i])):
        numbers[i][j] = numbers[i][j].astype('bool')





for i in range(0,len(image_paths)):
    print('Image: '+str(i+1)+'/'+str(len(image_paths)))
    #podesiti odgovarajuci path
    img = get_img(train_path+image_paths[i])

    start_indices = mark_indices(img.reshape(640*480))
    
    for start_indice in start_indices:
        img_d = get_image_from_indice(img,start_indice)
        nr = knn.predict(img_d)
        suma[i] = suma[i] + nr[0]
        suma[i] = int(suma[i])
        
upis(train_out, image_paths, suma)