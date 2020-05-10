from __future__ import print_function
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras import backend as K
K.backend.image_data_format=  'channels_last'  # for remote run
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
# set image dimension for Conv layer etc based on tensor flow or theano

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import sys
import glob
import argparse
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pickle
from helper_func import save_batch_info,smooth_curve,save_model_hyperparam,save_plots
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# If loading saved weihts, set to 1, else 0
load_mod = 0
save_model_weights = 1
save_best_weights = True # if 0, saves weights after last epoch
multi_gpu = 0
save_tr_val_stat_plots = 1 #save accuracy,loss plots for tr and val data

# get current filename without extension
fnm = os.path.basename(__file__)
fnm    = os.path.splitext(fnm)[0]
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")

bs_pth = './glucoma'
train_dir = bs_pth + '/no_resize/train'
validation_dir = bs_pth + '/no_resize/valid'
test_dir = bs_pth + '/no_resize/test'

pth_saved_model = bs_pth + '/saved_model' + '/' + dt_string
if not load_mod and not os.path.exists(pth_saved_model):
    os.makedirs(pth_saved_model)
pth_saved_model_json =  pth_saved_model +  '/' + fnm + '.json'
pth_saved_model_weights =  pth_saved_model +  '/' + fnm + '_weights.h5'

pth_saved_plots = bs_pth + '/saved_plots' +  '/' + dt_string
if not os.path.exists(pth_saved_plots):
    os.makedirs(pth_saved_plots)
# save flenames used in each batch of training nd validation data
save_batch_stats = 0
if save_batch_stats :
    pth_batch_tr = pth_saved_plots + '/' +  fnm + '_batch_tr.txt'
    pth_batch_vd = pth_saved_plots + '/' +  fnm + '_batch_vd.txt'


#************** Define the trainable layers in model****************
W = 299
H = 299
nc = 3
nclass = 2
resz_img = 1
# if running locally, resize to smaller size to be able to use larger batch size
if resz_img:
    W_rz = 128
    H_rz = 128
else:
    W_rz = W
    H_rz = H

if load_mod:
    # THIS IS SLOW,Avoid IT
    #model = load_model('./glucoma/saved_model/glucoma_v2.h5')
    # since models are saved in files as per time stamp, so run it with complete path
    json_file = open('C://ML//env//tf//glucoma//saved_model//06_02_2020_17_11//glucoma_v2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #Load weights saved from last epoch or the best weights
    model.load_weights('C://ML//env//tf//glucoma//saved_model//06_02_2020_17_11//glucoma_v2_weights.h5')
    model.summary()
else:
    def Resnet_finetune_model():
        Resnet = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(W_rz, H_rz, nc))
        model = models.Sequential()
        # add Resnet as the base( no need to specify input size here)
        model.add(Resnet)
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(nclass, activation='softmax'))
        return model

    model = Resnet_finetune_model()
    model.summary()

    # instantiate the Resnet base model
    conv_bs = model.get_layer('inception_resnet_v2')
    print("This is no of trainable weights in base Resnet "+ str(len(conv_bs.trainable_weights)))
    print("This is no of trainable weights in base Resnet and added dense layers before freezing"+ str(len(model.trainable_weights)))
    #conv_bs.trainable = False
    FREEZE_LAYERS = 2  
    # freeze the first this many layers for training\
    for layer in conv_bs.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in conv_bs.layers[FREEZE_LAYERS:]:
        layer.trainable = True
    print("This is no of trainable weights in base Resnet and added dense layers after freezing"+ str(len(model.trainable_weights)))
    for layer in conv_bs.layers:
        print(layer.name + "Trainable staus: " + str(layer.trainable) + "\n")
#class_weights = class_weight.compute_class_weight('balanced', np.unique(train_gen.classes),train_gen.classes)

# if not loadin model, train it
if not load_mod :

    # ************ create data generator, load data, and fit the model ********************
    batch_size = 32  # 2 with gtx 1650
    datagen_tr = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=40, width_shift_range=0.2,
                                    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, channel_shift_range=10,
                                    horizontal_flip=True, fill_mode='nearest')
    datagen_vd = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_gen = datagen_tr.flow_from_directory(train_dir, target_size=(W_rz, H_rz), batch_size=batch_size,
                                               class_mode='categorical', shuffle=True, interpolation="bilinear")
    vd_gen = datagen_vd.flow_from_directory(validation_dir, target_size=(W_rz, H_rz), batch_size=batch_size,
                                            class_mode='categorical', shuffle=False, interpolation="bilinear")
    # ***********************  print batch  data to files *************************
    if save_batch_stats:
        save_batch_info(train_gen, pth_batch_tr)
        save_batch_info(vd_gen, pth_batch_vd)
    # *******************************************
    nTrain = np.size(train_gen.classes)
    nVal = np.size(vd_gen.classes)
    epochs = 10
    steps_per_epoch_tr = int(nTrain / batch_size)
    steps_per_epoch_val = int(nVal / batch_size)
    # print learning rates
    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr

    lr_st = 1e-3
    optimizer = optimizers.Adam(lr=lr_st)
    lr_metric = get_lr_metric(optimizer)
    loss_func = 'categorical_crossentropy'
    if multi_gpu :
        parallel_model = multi_gpu_model(model, gpus=4)
        parallel_model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy', lr_metric])
        reduce_lr_cbk = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1,min_delta=1e-3, cooldown=0)
        # monitoring validation accuracy, and saving only the weights  for best epoch
        checkpointer = ModelCheckpoint(filepath=pth_saved_model_weights, monitor='val_acc',verbose=1, save_best_only=True)
        # show what happened in checkpoint
        callbacks_list = [reduce_lr_cbk, checkpointer]
        # callbacks_list = [checkpointer]
        if len(callbacks_list) == 0:
            callbacks_list = []
        # This `fit` call will be distributed on 8 GPUs.
        # Since the batch size is 256, each GPU will process 32 samples.
        parallel_model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = vd_gen,
                    validation_steps = steps_per_epoch_val,epochs=epochs,callbacks=callbacks_list,verbose=1)
    else:
        model.compile(optimizer=optimizer,loss=loss_func,metrics=['accuracy',lr_metric])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience = 5,min_lr= 1e-8,verbose=1,min_delta=1e-2, cooldown=2)
        # monitoring validation accuracy, and saving only the weights  for best epoch
        checkpointer = ModelCheckpoint(filepath= pth_saved_model_weights,monitor = 'val_acc',verbose = 1,save_best_only = save_best_weights)
        # show what happened in checkpoint
        callbacks_list = [checkpointer]
        #callbacks_list = [checkpointer,reduce_lr]
        if len(callbacks_list) == 0:
            callbacks_list = []
        # mdlfit=model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = vd_gen,
        #                     validation_steps = steps_per_epoch_val,epochs=epochs,class_weight=class_weights,callbacks=callbacks_list)
        mdlfit=model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = vd_gen,
                               validation_steps = steps_per_epoch_val,epochs=epochs,callbacks=callbacks_list,verbose=2)
    if save_model_weights:
        # write hyperparams to text file
        save_model_hyperparam(model, pth_saved_model,pth_batch_tr, pth_batch_vd, load_mod, save_model_weights, save_best_weights,
                              train_gen, vd_gen, batch_size, epochs, lr_st, optimizer, loss_func,
                              callbacks_list)
        model.save_weights(pth_saved_model_weights)
        model_json = model.to_json()
        with open(pth_saved_model_json, "w") as json_file:
            json_file.write(model_json)
            json_file.close()

#***********************************************************************
# ************plot training and validtion accuracy**********************
# if training the model, plot training and validation accuracy
if  not load_mod:
     if save_tr_val_stat_plots :
         save_plots(mdlfit, pth_saved_plots, fnm)
    #******************************************************************

# *************evalute on test data ************************************
datagen_tst = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = datagen_tst.flow_from_directory(test_dir,target_size=(W_rz, H_rz),batch_size=1,class_mode='categorical',shuffle=False,interpolation="bicubic")
# predict on test data, and save differences
filenames = test_gen.filenames
# Get number of test samples/images
nTest = len(filenames)
# Get 2 class probabilities
tst_pred = model.predict_generator(test_gen,steps=nTest)
#Find arg(max(prob))
test_pred = np.argmax(tst_pred,axis=1)
# Get truth labels
tst_lbls = test_gen.classes

plt.plot(range(nTest),tst_lbls,'b-',label = 'True Class Labels:Test')
plt.plot(range(nTest),test_pred,'g-',label = 'Predicted Class Lables"Test')
plt.title('Test Prediction')
plt.legend(loc='center right')
plt.savefig(pth_saved_plots  +  '/' + fnm + '_predict.png')
# Create 2-D array of True class and predicted class and save
tst_stat = np.vstack((tst_lbls, test_pred)).T
np.savetxt(pth_saved_plots  +  '/' + fnm + '_predict.txt', tst_stat)
dif = tst_lbls-test_pred
# count number of zeros( where true class natches predicted class)
print('test_acc_again:' + str(np.count_nonzero(dif==0)/nTest))


