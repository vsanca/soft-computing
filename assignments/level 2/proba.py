# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 21:28:02 2016

@author: Viktor
"""

import numpy as np
from sklearn.datasets import fetch_mldata

from skimage.io import imread, imshow, show
from skimage.morphology import opening, closing, remove_small_objects
from skimage.color import rgb2gray
from skimage.measure import label
from skimage.measure import regionprops
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier

#pomeranje slike u gornji levi ugao
def move(image,x,y):
    img = np.zeros((28,28))
    img[:(28-x),:(28-y)] = image[x:,y:]

    return img
    
#dopunjavanje slike do 28x28
def fill(image):
    if(np.shape(image)!=(28,28)):
        img = np.zeros((28,28))
        x = 28 - np.shape(image)[0]
        y = 28 - np.shape(image)[1]
        img[:-x,:-y] = image
        return img
    else:
        return image

#ucitavanje MNIST dataseta
def initialize():
    print('Initializing the solution and dataset')
    mnist = fetch_mldata('MNIST original')

    data = mnist.data>0
    data = data.astype('uint8')

    target = mnist.target

    obradjene = np.empty_like(data)

    #obrada podataka za trening
    for i in range(0,len(data)):
        l = label(data[i].reshape(28,28))
        r = regionprops(l)

        min_x = r[0].bbox[0]
        min_y = r[0].bbox[1]

        for j in range(1,len(r)):
            if(r[j].bbox[0]<min_x and r[j].bbox[1]<min_y):
                min_x = r[j].bbox[0]
                min_y = r[j].bbox[1]
                    
            
            #print(str(len(r))+', '+str(i))
            #imshow(r[ind].image)
            #show()
        #img = r[ind].image
        img = move(data[i].reshape(28,28),min_x,min_y)
        obradjene[i]= img.reshape(784,)
        #imshow(obradjene[i].reshape(28,28))
        #show()
        
    return obradjene, target

#postavljanje i obucavanje KNN
def fitKNN(data,target):
    print('Setting up the classifier')
    knn = KNeighborsClassifier(n_neighbors=1,weights='uniform',algorithm='auto', n_jobs=-1)
    knn.fit(data,target)
    return knn

#provera da li se regioni preklapaju
def check(c1,c2,par=15):
    if(abs(c1[0]-c2[0])<=par and abs(c1[1]-c2[1])<=par):
        return True
    else:
        return False

#spoj dva bliska regiona
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

    
def check2(indices,i):
    check = False
    for el in indices:
        if(el==i):
            check = True
            break
        
    return check

#izracunavanje sume
def getSum(path):
    image = imread(path)
    image = my_rgb2gray(image) > 0
    image = remove_small_objects(image,min_size=32)
    #imshow(image.astype('uint8'))
    #show()
    
    regions = label(image)
    labels = regionprops(regions)
    
    images = []
    indices = []
    
    for i in range(0,len(labels)):
        overlap = False
        for j in range(i+1,len(labels)):
            if(check(labels[i].centroid,labels[j].centroid,15)):
                overlap = True
                tmp = merge(image,labels[i].bbox,labels[j].bbox)
                images.append(tmp)
                indices.append(j)
        if(overlap==False and check2(indices,i)==False):
            images.append(labels[i].image)
        
    
    suma = 0
    for img in images:
        obrada = fill(np.array(img.astype('uint8')))
        #print(knn.predict(obrada.reshape(1,-1)))
        #imshow(obrada.astype('uint8'))
        #show()
        suma = suma + knn.predict(obrada.reshape(1,-1))
    
    return float(suma)
    
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

def my_rgb2gray(img_rgb):
    img_gray = np.ndarray((img_rgb.shape[0], img_rgb.shape[1]))
    img_gray = 0.5*img_rgb[:, :, 0] + 0.0*img_rgb[:, :, 1] + 0.5*img_rgb[:, :, 2]
    img_gray = img_gray.astype('uint8')
    return img_gray
        
obradjene, target = initialize()
knn = fitKNN(obradjene, target)

image_path = []
result = []
    
train_out = 'level-2-mnist-train/level-2-mnist-train/out.txt'
test_out = 'level-2-mnist-test/level-2-mnist-test/out.txt'

train_path = 'level-2-mnist-train/level-2-mnist-train/'
test_path = 'level-2-mnist-test/level-2-mnist-test/'

image_paths = ucitavanje(train_out)

for i in range(0,len(image_paths)):
    print(str(i+1)+'/'+str(len(image_paths)))
    s = getSum(train_path+image_paths[i])
    result.append(s)

upis(train_out,image_paths,result)




s = getSum(train_path+'images/img-82.png')

#94