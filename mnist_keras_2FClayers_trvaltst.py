
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
K.backend.image_data_format=  'channels_last'
seed = 7
np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# number of training samples
N1 = X_train.shape[0]  # same as  N1= X_train.shape and then N1 = N1[0]
N2 = X_test.shape[0]
h = X_train.shape[1]
w = X_train.shape[2]

num_pixels = h*w
x_train = X_train.reshape(N1, num_pixels).astype('float32') # shape is now (60000,784)
x_test = X_test.reshape(N2, num_pixels).astype('float32') # shape is now (10000,784)
x_train = x_train / 255
x_test = x_test / 255

y_train = to_categorical(y_train) #(10000,10): 10000 lables for 10 classes
y_test = to_categorical(y_test) # (10000,10): 10000 lables for 10 classes
num_classes = y_test.shape[1]
# load  the model
model = load_model('mnist_keras_2FClayers.h5')

trn=model.fit(x_train, y_train, validation_split=0.15, epochs=20, batch_size=200, verbose=2,shuffle=True)

# Final evaluation of the model on test data
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Accuracy: %.2f%%" % (scores[1]*100))
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# Note: as test set is same as validation, validation accuracy matched the test accuracy here

#plot training accuracy and loss
trn_acc = trn.history['accuracy']
trn_loss = trn.history['loss']
epochs = range(len(trn_acc))
plt.plot(epochs,trn_acc,'bo',label = 'Train accuracy')
plt.plot(epochs,trn_loss,'b',label = 'Train Loss')
plt.show()

# plot validation accuracy and loss
val_acc = trn.history['val_accuracy']
val_loss = trn.history['val_loss']
epochs = range(len(trn_acc))
plt.plot(epochs,val_acc,'bo',label = 'Train accuracy')
plt.plot(epochs,val_loss,'b',label = 'Train Loss')
plt.show()
