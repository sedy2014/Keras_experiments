
# wrt version4,I use subset 1, where training dats is more balanced
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator, image,load_img
from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers
from keras.models import load_model
from keras.models import model_from_json
# set image dimension for Conv layer etc based on tensor flow or theano
K.set_image_dim_ordering('tf')
from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.callbacks import ModelCheckpoint
import os
import sys
import glob
import argparse
#K.clear_session()
from keras.callbacks import ReduceLROnPlateau
#************** Define the trainable layers in model****************
# input shape for VGG16, modified for cropped data saved as 512 by 512
W = 299
H = 299
nc = 3
nclass = 2
load_mod = 1
if load_mod:
    #model = load_model('diabetic_v9.h5')
    json_file = open("diabetic_v9.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("diabetic_v9_weights.h5")
    model.summary()
else:
    def Resnet_finetune_model():
        Resnet = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(W, H, nc))
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
    FREEZE_LAYERS = 2  # freeze the first this many layers for training\
    #conv_bs.summary()
    # for layer in conv_bs.layers:
    #     layer.trainable = False
    # for layer in conv_bs.layers[7:10]:
    #     layer.trainable = True
    # for layer in conv_bs.layers[11:14]:
    #     layer.trainable = True
    # for layer in conv_bs.layers[15:18]:
    #     layer.trainable = True
    for layer in conv_bs.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in conv_bs.layers[FREEZE_LAYERS:]:
        layer.trainable = True
    print("This is no of trainable weights in base Resnet and added dense layers after freezing"+ str(len(model.trainable_weights)))
    for layer in conv_bs.layers:
        print(layer.name + "Trainable staus: " + str(layer.trainable) + "\n")


    #************************************************************
    #************ create data generator, load data, and fit the model ********************

bs_pth = './Glaucoma-Normal_for_Sid/no_resize'
train_dir = bs_pth + '/train'
validation_dir = bs_pth + '/valid'
test_dir = bs_pth + '/test'

batch_size = 8
datagen_tr  = ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,channel_shift_range=10,horizontal_flip=True,fill_mode='nearest')
datagen_vd  = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = datagen_tr.flow_from_directory(train_dir,target_size=(W, H),batch_size=batch_size,class_mode='categorical',shuffle=True,interpolation="bilinear")
vd_gen = datagen_vd.flow_from_directory(validation_dir,target_size=(W, H),batch_size=batch_size,class_mode='categorical',shuffle=False,interpolation="bilinear")

#class_weights = class_weight.compute_class_weight('balanced', np.unique(train_gen.classes),train_gen.classes)

nTrain = np.size(train_gen.classes)
nVal = np.size(vd_gen.classes)
epochs = 40
steps_per_epoch_tr = int(nTrain/ batch_size)
steps_per_epoch_val =  int(nVal/batch_size)

if not load_mod :
    model.compile(optimizer=optimizers.Adam(lr=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience = 5,min_lr= 1e-7,verbose=1)
    # monitoring validation accuracy, and saving only the weights  for best epoch
    checkpointer = ModelCheckpoint(filepath="diabetic_v9_bestWeights.h5",monitor = 'val_acc',verbose=1,save_best_only=True)

    # show what happened in checkpoint
    callbacks_list = [checkpointer]
    # mdlfit=model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = vd_gen,
    #                     validation_steps = steps_per_epoch_val,epochs=epochs,class_weight=class_weights,callbacks=callbacks_list)
    mdlfit=model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = vd_gen,
                    validation_steps = steps_per_epoch_val,epochs=epochs,callbacks=callbacks_list)
    if not load_mod:
        #model.save('diabetic_v9.h5')
        model.save_weights('diabetic_v9_weights.h5')
        model_json = model.to_json()
        with open('diabetic_v9.json', "w") as json_file:
            json_file.write(model_json)
            json_file.close()

#***********************************************************************

# ************plot training and validtion accuracy**********************
def smooth_curve(data,alpha):
    smooth_d = []
    for point in data:
        if smooth_d:
            prev = smooth_d[-1]
            smooth_d.append((prev * alpha) + point * (1 - alpha))
        else:
            smooth_d.append(point)

    return smooth_d

if not load_mod :
    trn_acc = mdlfit.history['acc']
    val_acc = mdlfit.history['val_acc']
    epochs = range(len(trn_acc))
    plt.plot(epochs,trn_acc,'b-',label = 'Train accuracy')
    plt.plot(epochs,val_acc,'g-',label = 'Valid accuracy')
    plt.title('Accuracy')
    plt.legend(loc='center right')
    plt.savefig('./diabeticRetino/diabetic_v9_acc.png')
    plt.show()
    plt.plot(epochs,smooth_curve(trn_acc,0.8),'b-',label = 'Smoothed Train accuracy')
    plt.plot(epochs,smooth_curve(val_acc,0.8),'g-',label = 'Smoothed Valid accuracy')
    plt.title('Smoothed Accuracy')
    plt.legend(loc='center right')
    plt.savefig('./diabeticRetino/diabetic_v9_smooth_acc.png')
    plt.show()
    #plot training and validtion loss
    trn_loss = mdlfit.history['loss']
    val_loss = mdlfit.history['val_loss']
    epochs = range(len(trn_loss))
    plt.plot(epochs,trn_loss,'b-',label = 'Train Loss')
    plt.plot(epochs,val_loss,'g-',label = 'Valid Loss')
    plt.title('Loss')
    plt.legend(loc='center right')
    plt.savefig('./diabeticRetino/diabetic_v9_loss.png')
    plt.show()
    plt.plot(epochs,smooth_curve(trn_loss,0.8),'b-',label = 'Smoothed Train Loss')
    plt.plot(epochs,smooth_curve(val_loss,0.8),'g-',label = 'Smoothed Valid Loss')
    plt.title('Smoothed Loss')
    plt.legend(loc='center right')
    plt.savefig('./diabeticRetino/diabetic_v9_smooth_loss.png')
    plt.show()

    #******************************************************************

# *************evalute on test data ************************************
# methods 1 aS IN PAPER

def get_files(path):
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, '*'))
        elif path.find('*') > 0:
            files = glob.glob(path)
        else:
            files = [path]

        files = [f for f in files if f.endswith('png') or f.endswith('png')]
        if not len(files):
            sys.exit('No images found by the given path!')
        return files
print("***********Test data :predict class 0*************************************")
files = get_files(test_dir + '/0')
cls_list = ['Normal','GandGS']
print(cls_list)
# 2-d numpy arrray  of probabibility of each class for each file
pred_c0 = np.empty((0, nclass))
sum_true_class = np.size(files)

for f in files:
    img = image.load_img(f, target_size=(299,299))
    if img is None:
        continue
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0] # [ [a,b]] so needs .x[0]
    pred_c0 = np.append(pred_c0,[pred],axis=0)
    # index of max prob
    indxmx = np.argmax(pred)
    if indxmx != 0:
        sum_true_class = sum_true_class - 1
    top_inds = pred.argsort()[::-1][:5]
    print(f)
    for i in top_inds:
        print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
np.savetxt('./diabeticRetino/diabetic_v9_predict_c0.txt', pred_c0)
print("calss 0 accuracy  = " + str( (sum_true_class/np.size(files))* 100 )  + '%')

print("***********Test data :predict class 1*************************************")

files = get_files(test_dir + '/1')
pred_c1 = np.empty((0, nclass))
sum_true_class = np.size(files)
for f in files:
    img = image.load_img(f, target_size=(299,299))
    if img is None:
        continue
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0]  # net_final or net\n",
    # y = 0
    # ev = model.evaluate(x,y,batch_size= 1)
    pred_c1 = np.append(pred_c1, [pred], axis=0)
    # index of max prob
    indxmx = np.argmax(pred)
    if indxmx != 1:
        sum_true_class = sum_true_class - 1
    top_inds = pred.argsort()[::-1][:5] # gives indices [0,1]
    print(f)
    # print probabibily and coressponding class name
    for i in top_inds:
        print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
np.savetxt('./diabeticRetino/diabetic_v9_predict_c1.txt', pred_c1)
print("calss 1 accuracy  = " + str( (sum_true_class/np.size(files))* 100 )  + '%')


test_gen = datagen_vd.flow_from_directory(test_dir,target_size=(W, H),batch_size=1,class_mode='categorical',shuffle=False,interpolation="bicubic")
# predict on test data, and save differences
filenames = test_gen.filenames
nTest = len(filenames)
tst_pred = model.predict_generator(test_gen,steps=nTest)
test_pred = np.argmax(tst_pred,axis=1)
tst_lbls = test_gen.classes

plt.plot(range(nTest),tst_lbls,'b-',label = 'True Class Labels:Test')
plt.plot(range(nTest),test_pred,'g-',label = 'Predicted Class Lables"Test')
plt.title('Test Prediction')
plt.legend(loc='center right')
plt.savefig('./diabeticRetino/diabetic_v9_predict.png')

tst_stat = np.vstack((tst_lbls, test_pred)).T
np.savetxt('./diabeticRetino/diabetic_v9_predict.txt', tst_stat)
dif = tst_lbls-test_pred
# count number of zeros( where true class natches predicted class)
print('test_acc_again:' + str(np.count_nonzero(dif==0)/nTest))


