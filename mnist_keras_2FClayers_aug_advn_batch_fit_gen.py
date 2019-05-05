
#MNIST prediction with FC layers, and IMAGE Data generator 

from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
# set image dimension for Conv layer etc based on tensor flow or theano
K.set_image_dim_ordering('tf')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load (downloaded if needed to : C:\Users\sidha\.keras\datasets\mnist.npz) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# separate data into train and validation
from sklearn.model_selection import train_test_split
# Split the data
valid_per = 0.15
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_per, shuffle= True)

N1 = X_train.shape[0]  # training size
N2 = X_test.shape[0]   # test size
N3 = X_valid.shape[0]  # valid size
h = X_train.shape[1]
w = X_train.shape[2]


num_pixels = h*w
# reshape N1 samples to num_pixels
#x_train = X_train.reshape(N1, num_pixels).astype('float32') # shape is now (51000,784)
#x_test = X_test.reshape(N2, num_pixels).astype('float32') # shape is now (9000,784)


y_train = np_utils.to_categorical(y_train) #(51000,10): 10000 lables for 10 classes
y_valid = np_utils.to_categorical(y_valid)  #(9000,10): 9000 labels for 10 classes
y_test = np_utils.to_categorical(y_test) # (10000,10): 10000 lables for 10 classes
 
num_classes = y_test.shape[1]

def baseline_model():
# create model
    model = Sequential()
	# flatten input to (N1,w*h) as fit_generator expects (N1,w*h), but dont' have x,y as inputs(so cant reshape)
    model.add(Flatten(input_shape=(h,w,1)))
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    # Define output layer with softmax function
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
model.summary()

batch_size = 200
epochs = 20
steps_per_epoch_tr = int(N1/ batch_size) # 51000/200
steps_per_epoch_val =  int(N3/batch_size) 

# reshape to be [samples][width][height][ channel] for ImageData Gnerator->datagen.flow
x_t = X_train.reshape(N1, w, h, 1).astype('float32')
x_v = X_valid.reshape(N3, w, h, 1).astype('float32')

# define data preparation
x_t = x_t/255
x_v = x_v/255

datagen_tr = ImageDataGenerator(featurewise_center = True,featurewise_std_normalization=True,width_shift_range=0.1,height_shift_range=0.1)
datagen_vd = ImageDataGenerator(featurewise_center = True,featurewise_std_normalization=True)
datagen_tr.fit(x_t,augment=True)
datagen_vd.fit(x_v,augment=True)
train_gen = datagen_tr.flow(x_t, y_train, batch_size=batch_size)
valid_gen = datagen_vd.flow(x_v,y_valid, batch_size=batch_size)
mdlfit=model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = valid_gen,
                    validation_steps = steps_per_epoch_val,epochs=epochs)

#plot training accuracy and loss
trn_acc = mdlfit.history['acc']
trn_loss = mdlfit.history['loss']
epochs = range(len(trn_acc))
plt.plot(epochs,trn_acc,'bo',label = 'Train accuracy')
plt.plot(epochs,trn_loss,'b',label = 'Train Loss')


# plot validation accuracy and loss
val_acc = mdlfit.history['val_acc']
val_loss = mdlfit.history['val_loss']
epochs = range(len(trn_acc))
plt.plot(epochs,val_acc,'bo',label = 'Valid accuracy')
plt.plot(epochs,val_loss,'b',label = 'Valid Loss')

x_test = X_test.reshape(N2, w, h, 1).astype('float32')
# Final evaluation of the model on test data( will predict to classes and also give error)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Accuracy: %.2f%%" % (scores[1]*100))
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# configure batch size and retrieve one batch of images
# save in C:/ML/env/tf/MNIST/aug_images
aug_path = "./MNIST/aug_images"
if not os.path.isdir("./MNIST/aug_images"):
    os.makedirs(aug_path)

i = 0
for x_batch, y_batch in datagen.flow(x_t, y_train, batch_size=200,save_to_dir=aug_path, save_prefix='aug', save_format='png'):
    # create a grid of 3x3 images
    #for i in range(0, 9):
	    #pyplot.subplot(330 + 1 + i)
	    #pyplot.imshow(x_batch[i].reshape(h, w), cmap=pyplot.get_cmap('gray'))
	    # show the plot
	   # pyplot.show()
    i = i + 1
    if i==1 :
        break







