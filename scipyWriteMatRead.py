
from __future__ import print_function
import numpy as np
import scipy.io
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img,array_to_img, img_to_array
from tensorflow.keras import backend as K

K.backend.image_data_format=  'channels_last'

x1= np.array([[1, 2], [3, 4]])
x2 = np.array([[5, 6], [7, 8]])
x3= np.array(([9, 10], [11, 12]))
x4 = np.dstack((x1,x2))
x5 =  np.dstack((x4,x3))
print(x5)
x6 = x5
scipy.io.savemat('./MNIST/test/scipyWriteMatread_v1.mat', dict(x1=x1, x2=x2,x3=x3,x5=x5))
scipy.io.savemat('./MNIST/test/scipyWriteMatread_v2.mat', mdict={'arr': x6})

x_t = x5.reshape(3, 2, 2, 1)
datagen_rescale = ImageDataGenerator(rescale=1)

print('\n******** datagen_rescale 1st column************\n')
n=0
for x_batch1 in datagen_rescale.flow(x_t, y=None, shuffle = False,batch_size=3):
    a= x_batch1.reshape(2,2,3)
    print(a[:,:,0])
    print(a[:,:,1])
    print(a[:,:,2])
    scipy.io.savemat('./MNIST/test/scipyWriteMatread_v3.mat', mdict={'arr': a})
    n = n + 1
    if n == 1:
        break
