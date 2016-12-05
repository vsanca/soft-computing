# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:27:22 2016

@author: Viktor
"""

import numpy as np
from sklearn.datasets import fetch_mldata
from matplotlib import pyplot as plt


from skimage.io import imread
from skimage.io import imshow
from skimage.morphology import opening, closing
from scipy import ndimage


from sklearn.neighbors import KNeighborsClassifier


#ucitavanje MNIST dataseta
mnist = fetch_mldata('MNIST original')
print(mnist.data.shape)
print(mnist.target.shape)
print(np.unique(mnist.target))


img = 255-mnist.data[12345]
img = img.reshape(28,28)

plt.imshow(-img, cmap='Greys')

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


test = numbers[0][123]

res = numbers[0][123] == numbers[0][124]

percent_hit = np.count_nonzero(res) / 784.0


representative_number = [0]*10

for j in range(0,10):
    representative_number[j] = np.zeros(np.shape(numbers[j][0]), dtype='float')
    for i in range(0,len(numbers[j])):
        representative_number[j] = representative_number[j] + numbers[j][i]

    representative_number[j] = (representative_number[j])/len(numbers[j])


def processing(path):
    img = imread(path)
    gray = rgb2gray(img)

    binary = 1 - (gray > 0.5)
    binary = closing(binary)
    binary = opening(binary)

    labeled, nr_objects = ndimage.label(binary)
    return nr_objects

def poklapanje(niz1, niz2):
    mera_poklapanja = 0.0
    for i in range(0,len(niz1)):
        if(niz1[i]==niz2[i]):
            mera_poklapanja = mera_poklapanja + 1
            
    return mera_poklapanja/len(niz1)
    
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
    
def find_number(image28_28):
    mmx = [0]*10
    for i in range(0,10):
        for j in range(0,len(numbers[i])):
            res = binarize(image28_28) == binarize(numbers[i][j])
            if(np.count_nonzero(res)>mmx[i]):
                mmx[i]=np.count_nonzero(res)
    
    return max_idx(mmx)
    
def max_idx(lista):
    mx = max(lista)
    for i in range(0,len(lista)):
        if(lista[i]==mx):
            return i
            
    return -1
    
image_path = []
result = []
    
in_path = 'level-1-mnist-train/level-1-mnist/out.txt'
out_path = 'level-1-mnist-test/level-1-mnist-test/out.txt'

train_path = 'level-1-mnist-train/level-1-mnist/'
test_path = 'level-1-mnist-test/level-1-mnist-test/'

image_paths = ucitavanje(out_path)

#knn = KNeighborsClassifier()
knn = KNeighborsClassifier(n_neighbors=2000,weights='distance',algorithm='auto',n_jobs=-1)
knn.fit(mnist.data,mnist.target)

suma = [0]*len(image_paths)

for i in range(0,len(image_paths)):
    print('Image'+str(i+1)+'/'+str(len(image_paths)))
    img = get_img(test_path+image_paths[i])

    start_indices = mark_indices(img.reshape(640*480))
    
    for start_indice in start_indices:
        img_d = get_image_from_indice(img,start_indice)
        #nr = find_number(img_d)
        nr = knn.predict(img_d)
        suma[i] = suma[i] + nr[0]
        suma[i] = int(suma[i])


for i in range(0,len(suma)):
    suma[i] = float(suma[i])
        
upis(out_path, image_paths, suma)





image28_28 = img_d
mmx = [0]*10
for i in range(0,10):
    for j in range(0,len(numbers[i])):
        res = image28_28 == numbers[i][j]
        if(np.count_nonzero(res)>mmx[i]):
            mmx[i]=np.count_nonzero(res)


    
total = np.zeros(784, dtype='float')
for i in range(0,10):
    total = total + representative_number[i]
        
img = representative_number[4]
img = img.reshape(28,28)
plt.imshow(img, cmap='Greys')

check = numbers[5][123]


suma = [0]*10
klasa = 0

for j in range(0,10):
    for i in range(0,len(check)):
        if(check[i]!=0):
            suma[j] = suma[j] + representative_number[j][i]
    if(suma[j]>max(suma)):
        klasa = j

print(klasa)
