
# coding: utf-8

# In[ ]:


# MNISt dataSet using Sequential model with 2 FC layers


# In[ ]:


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


# In[ ]:


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# In[ ]:


# load (downloaded if needed to : C:\Users\sidha\.keras\datasets\mnist.npz) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[ ]:


print(X_train.shape) # X_train.shape result is a tuple
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# number of training samples
N1 = X_train.shape[0]  # same as  N1= X_train.shape and then N1 = N1[0]
N2 = X_test.shape[0]  
h = X_train.shape[1]
w = X_train.shape[2]


# In[ ]:


# Get the shape of data
print(X_train[0].shape)
print(X_test[0].shape)


# In[ ]:


tmp = 220
# plot 4 images as gray scale
#211 is equivalent to nrows=2, ncols=1, plot_number=1. ~ to matlab subplot(2,1)
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()


# In[ ]:


#Baseline Model with Multi-Layer Perceptrons

#For a multi-layer perceptron model we must reduce the images down into a vector of pixels. 
#In this case the 28×28 sized images will be 784 pixel input values.
num_pixels = h*w
# reshape N1 samples to num_pixels
x_train = X_train.reshape(N1, num_pixels).astype('float32') # shape is now (60000,784)
x_test = X_test.reshape(N2, num_pixels).astype('float32') # shape is now (10000,784)


# In[ ]:


x_train = x_train / 255
x_test = x_test / 255


# In[ ]:


y_test[0]


# In[ ]:


#Finally, the output variable is an integer from 0 to 9. This is a multi-class classification problem. 10 digits 
# classified to 10 classes
#As such, it is good practice to use a one hot encoding of the class values,
#transforming the vector of class integers into a binary matrix.

#We can easily do this using the built-in np_utils.to_categorical() helper function in Keras.
y_train = to_categorical(y_train) #(10000,10): 10000 lables for 10 classes
y_test =  to_categorical(y_test) # (10000,10): 10000 lables for 10 classes
num_classes = y_test.shape[1]


# In[ ]:


y_test[0]  # now, digit N is being repesented as [0 0 .. 1 ..0] where 1 is at index N


# In[ ]:


def baseline_model():
	# create Sequential model : linear stack of layers.
	model = Sequential()
    # Define input layer which with same number of neurons as there are inputs (784), but can have less or more neurons 
    # use RELU for this hidden layer
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    # Define output layer with softmax function, now #neurons must match number of op classes
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
    #  use ADAm optimizer and  Logarithmic loss or  categorical_crossentropy Loss
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# if we need to modify default parameters in optimizer , we do it this way
#      from tensorflow.keras import optimizers
#      adam = optimizers.ADAM(lr=0.001, beta_1=0.9, beta_2=0.999)
#      model.compile(  ,optimizer = adam)


# In[ ]:


model = baseline_model()
model.summary()


# In[ ]:


# save the model
#model.save("mnist_keras_2FClayers.h5")


# In[ ]:


# Train the model
# test data is used as validation data
#  A verbose value of 2 is used to reduce the output to one line for each training epoch.
trn=model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)


# In[ ]:


# Final evaluation of the model on test data
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Accuracy: %.2f%%" % (scores[1]*100))
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# Note: as test set is same as validation, validation accuracy matched the test accuracy here


# In[ ]:


#plot training accuracy and loss
trn_acc = trn.history['accuracy']
trn_loss = trn.history['loss']
epochs = range(len(trn_acc))
plt.plot(epochs,trn_acc,'bo',label = 'Train accuracy')
plt.plot(epochs,trn_loss,'b',label = 'Train Loss')
plt.show()

# In[ ]:


# plot validation accuracy and loss
val_acc = trn.history['val_accuracy']
val_loss = trn.history['val_loss']
epochs = range(len(trn_acc))
plt.plot(epochs,val_acc,'bo',label = 'Train accuracy')
plt.plot(epochs,val_loss,'b',label = 'Train Loss')
plt.show()


# In[ ]:


#Get classification report
y_p = model.predict_classes(x_test)
y_p = to_categorical(y_p)
from sklearn.metrics import classification_report
target_nms = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test,y_p,target_names=target_nms))
print('Done')
