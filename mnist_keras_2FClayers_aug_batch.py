
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
# set image dimension for Conv layer etc based on tensor flow or theano
K.backend.image_data_format=  'channels_last'
seed = 7
np.random.seed(seed)

# load (downloaded if needed to : C:\Users\sidha\.keras\datasets\mnist.npz) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
N1 = X_train.shape[0]  
N2 = X_test.shape[0]  
h = X_train.shape[1]
w = X_train.shape[2]
num_pixels = h*w
# reshape N1 samples to num_pixels
x_train = X_train.reshape(N1, num_pixels).astype('float32') # shape is now (60000,784)
x_test = X_test.reshape(N2, num_pixels).astype('float32') # shape is now (10000,784)

y_train = to_categorical(y_train) #(60000,10): 10000 lables for 10 classes
y_test = to_categorical(y_test) # (10000,10): 10000 lables for 10 classes
num_classes = y_test.shape[1]

def baseline_model():
# create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    # Define output layer with softmax function
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
model.summary()
batch_size = 200
epochs = 2
max_batches = 2 * len(x_train) / batch_size # 2*60000/200

# reshape to be [samples][width][height][ channel] for ImageDataGenerator
x_t = X_train.reshape(N1, w, h, 1).astype('float32')
datagen = ImageDataGenerator(rescale= 1./255)
train_gen = datagen.flow(x_t, y_train, batch_size=batch_size)
for e in range(epochs):
    batches = 0
    for x_batch, y_batch in train_gen:
    # x_batch is of size [batch_sz,w,h,ch]: resize to [bth_sz,pixel_sz]: (200,28,28,1)-> (200,784)
    # for model.fit
        x_batch = np.reshape(x_batch, [-1, num_pixels]) 
        # do validation split		
        model.fit(x_batch, y_batch,validation_split=0.15,verbose=0)
        batches += 1
        print("Epoch %d/%d, Batch %d/%d" % (e+1, epochs, batches, max_batches))
        if batches >= max_batches:
        # we need to break the loop by hand because
        # the generator loops indefinitely
            break
 
x_test = x_test.reshape(N2, w, h, 1)
# Final evaluation of the model on test data( will predict to classes and also give error)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Accuracy: %.2f%%" % (scores[1]*100))
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

