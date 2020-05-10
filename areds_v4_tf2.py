from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as k
k.backend.image_data_format=  'channels_last'  # for remote run
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers,models
from keras.models import  model_from_json
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
#import cv2
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import multi_gpu_model
# Save data from original directory to train,test etc location
#bs_pth = 'C://ML//env//tf//areds//input'
bs_pth = './/pycharm_areds3//areds//input'
train_dir = bs_pth + '//train'
validation_dir = bs_pth + '//valid'
test_dir = bs_pth + '//test'
output = bs_pth + '//output'
tr_per = 0.8
val_per = 0.2
tst_per = 1 - tr_per - val_per
mv_cpy=0
shuffle = 'True'
class_nms = ['Control','Control_Questionable_1','Control_Questionable_2','Control_Questionable_3','Control_Questionable_4','Large_Drusen','Large_Drusen_Questionable_1',
                       'Large_Drusen_Questionable_2','Large_Drusen_Questionable_3','Questionable_AMD','NV_AMD','GA','Both_NV_AMD_and_GA']

#************** Define the trainable layers in model****************
W = 100  #299
H = 100  # 299
nc = 3
num_pixels = W*H
nclass = len(class_nms)
# If loading saved weihts, set to 1, else 0
load_mod = 0
multi_gpu = 0
if load_mod:
    # This is really slow, so avoid it
    #model = load_model('./areds/saved_model/areds_v3.h5') C:\ML\env\tf\areds\saved_model
    json_file = open("./areds/saved_model/areds_v3.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #Load weights saved from last epoch
    #model.load_weights("./areds/saved_model/areds_v3_weights.h5")
    # load best weights
    model.load_weights("./areds/saved_model/areds_v3_bestWeights.h5")
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
    def baseline_model():
        # create model
        model = models.Sequential()
        # flatten input to (N1,w*h) as fit_generator expects (N1,w*h), but dont' have x,y as inputs(so cant reshape)
        model.add(layers.Flatten(input_shape=(H, W, 3)))
        model.add(layers.Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
        # Define output layer with softmax function
        model.add(layers.Dense(nclass, kernel_initializer='normal', activation='softmax'))
        return model

    #model = Resnet_finetune_model()
    model = baseline_model()
    model.summary()

    # instantiate the Resnet base model
    #conv_bs = model.get_layer('inception_resnet_v2')
   # print("This is no of trainable weights in base Resnet "+ str(len(conv_bs.trainable_weights)))
   # print("This is no of trainable weights in base Resnet and added dense layers before freezing"+ str(len(model.trainable_weights)))
    #conv_bs.trainable = False
    # FREEZE_LAYERS = 2
    # # freeze the first this many layers for training\
    # for layer in conv_bs.layers[:FREEZE_LAYERS]:
    #     layer.trainable = False
    # for layer in conv_bs.layers[FREEZE_LAYERS:]:
    #     layer.trainable = True
   # print("This is no of trainable weights in base Resnet and added dense layers after freezing"+ str(len(model.trainable_weights)))
   # for layer in conv_bs.layers:
       # print(layer.name + "Trainable staus: " + str(layer.trainable) + "\n")


batch_size = 2
#datagen_tr  = ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,channel_shift_range=10,horizontal_flip=True,fill_mode='nearest')
datagen_tr  = ImageDataGenerator(preprocessing_function=preprocess_input)
datagen_vd  = ImageDataGenerator(preprocessing_function=preprocess_input)
train_gen = datagen_tr.flow_from_directory(train_dir,target_size=(W, H),batch_size=batch_size,class_mode='categorical',shuffle=True,interpolation="bilinear")
vd_gen = datagen_vd.flow_from_directory(validation_dir,target_size=(W, H),batch_size=batch_size,class_mode='categorical',shuffle=False,interpolation="bilinear")

#class_weights = class_weight.compute_class_weight('balanced', np.unique(train_gen.classes),train_gen.classes)

nTrain = np.size(train_gen.classes)
nVal = np.size(vd_gen.classes)
epochs = 2
steps_per_epoch_tr = int(nTrain/ batch_size)
steps_per_epoch_val =  int(nVal/batch_size)

# if not loadin model, train it
if not load_mod :
    # print learning rates
    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr
    lr_st = 1e-5
    optimizer = optimizers.Adam(lr=lr_st)
    lr_metric = get_lr_metric(optimizer)
    if multi_gpu :
        parallel_model = multi_gpu_model(model, gpus=4)
        parallel_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', lr_metric])
        reduce_lr_cbk = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1,
                                          min_delta=1e-3, cooldown=0)

        # monitoring validation accuracy, and saving only the weights  for best epoch
        checkpointer = ModelCheckpoint(filepath="./areds/saved_model/areds_v3_bestWeights.h5", monitor='val_acc',
                                       verbose=1, save_best_only=True)
        tb_cbk = TensorBoard(log_dir='./areds/saved_model', histogram_freq=0, write_graph=True, write_images=True)
        # show what happened in checkpoint
        callbacks_list = [reduce_lr_cbk, checkpointer]
        # callbacks_list = [checkpointer]
        # callbacks_list = [reduce_lr_cbk, checkpointer,tb_cbk]
        # mdlfit=model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = vd_gen,
        #                     validation_steps = steps_per_epoch_val,epochs=epochs,class_weight=class_weights,callbacks=callbacks_list)
        # This `fit` call will be distributed on 8 GPUs.
        # Since the batch size is 256, each GPU will process 32 samples.
        parallel_model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = vd_gen,
                    validation_steps = steps_per_epoch_val,epochs=epochs,callbacks=callbacks_list,verbose=1)
    else:
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', lr_metric])
            reduce_lr_cbk = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,  verbose=1,
                          min_delta=1e-3,cooldown=1)

            # monitoring validation accuracy, and saving only the weights  for best epoch
            checkpointer = ModelCheckpoint(filepath="./areds/saved_model/areds_v3_bestWeights.h5",monitor = 'val_acc',verbose = 1,save_best_only = True)
            tb_cbk = TensorBoard(log_dir='./areds/saved_model', histogram_freq=0, write_graph=True, write_images=True)
            # show what happened in checkpoint
            callbacks_list = [reduce_lr_cbk,checkpointer]
            #callbacks_list = [checkpointer]
            #callbacks_list = [reduce_lr_cbk, checkpointer,tb_cbk]
            # mdlfit=model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = vd_gen,
            #                     validation_steps = steps_per_epoch_val,epochs=epochs,class_weight=class_weights,callbacks=callbacks_list)
            mdlfit=model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = vd_gen,
                    validation_steps = steps_per_epoch_val,epochs=epochs,callbacks=callbacks_list,verbose=1)
    if not load_mod:
        #model.save('./glucoma/saved_model/areds_v3.h5')
        model.save_weights('./areds/saved_model/areds_v3_weights.h5')
        model_json = model.to_json()
        with open('./areds/saved_model/areds_v3.json', "w") as json_file:
            json_file.write(model_json)
            json_file.close()

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

# if training the model, plot training and validation accuracy
if not load_mod :
    trn_acc = mdlfit.history['acc']
    val_acc = mdlfit.history['val_acc']
    lr_ep = mdlfit.history['lr']
    epochs = range(len(trn_acc))
    plt.plot(epochs,trn_acc,'b-',label = 'Train accuracy')
    plt.plot(epochs,val_acc,'g-',label = 'Valid accuracy')
    plt.title('Accuracy')
    plt.legend(loc='center right')
    plt.savefig('./areds/saved_plots/areds_v3_acc.png')
    plt.show()
    plt.plot(epochs,smooth_curve(trn_acc,0.8),'b-',label = 'Smoothed Train accuracy')
    plt.plot(epochs,smooth_curve(val_acc,0.8),'g-',label = 'Smoothed Valid accuracy')
    plt.title('Smoothed Accuracy')
    plt.legend(loc='center right')
    plt.savefig('./areds/saved_plots/areds_v3_smooth_acc.png')
    #plt.show()
    #plot training and validtion loss
    trn_loss = mdlfit.history['loss']
    val_loss = mdlfit.history['val_loss']
    epochs = range(len(trn_loss))
    plt.plot(epochs,trn_loss,'b-',label = 'Train Loss')
    plt.plot(epochs,val_loss,'g-',label = 'Valid Loss')
    plt.title('Loss')
    plt.legend(loc='center right')
    plt.savefig('./areds/saved_plots/areds_v3_loss.png')
    #plt.show()
    plt.plot(epochs,smooth_curve(trn_loss,0.8),'b-',label = 'Smoothed Train Loss')
    plt.plot(epochs,smooth_curve(val_loss,0.8),'g-',label = 'Smoothed Valid Loss')
    plt.title('Smoothed Loss')
    plt.legend(loc='center right')
    plt.savefig('./areds/saved_plots/areds_v3_smooth_loss.png')
    #plt.show()

    plt.plot(epochs, lr_ep, 'b-', label='Learning Rate')
    plt.title('Learning_Rt')
    plt.legend(loc='center right')
    plt.savefig('./glucoma/saved_plots/glucoma_v1_lr.png')
    #plt.show()
    #np.savetxt('./areds/saved_plots/areds_v3_tracc.txt', trn_acc)
    #np.savetxt('./areds/saved_plots/areds_v3_valacc.txt', val_acc)
    #np.savetxt('./areds/saved_plots/areds_v3_trloss.txt', trn_loss)
    #np.savetxt('./areds/saved_plots/areds_v3_valoss.txt', val_loss)
    np.savetxt('./areds/saved_plots/areds_v3_all.txt', np.vstack(((trn_acc,val_acc),(trn_loss,val_loss))).T)
    np.savetxt('./areds/saved_plots/areds_v3_lr.txt', lr_ep)
    #******************************************************************
# *************evalute on test data ************************************

# test_gen = datagen_vd.flow_from_directory(test_dir,target_size=(W, H),batch_size=1,class_mode='categorical',shuffle=False,interpolation="bicubic")
# # predict on test data, and save differences
# filenames = test_gen.filenames
# # Get number of test samples/images
# nTest = len(filenames)
# # Get 2 class probabilities
# tst_pred = model.predict_generator(test_gen,steps=nTest)
# #Find arg(max(prob))
# test_pred = np.argmax(tst_pred,axis=1)
# # Get truth labels
# tst_lbls = test_gen.classes
#
# plt.plot(range(nTest),tst_lbls,'b-',label = 'True Class Labels:Test')
# plt.plot(range(nTest),test_pred,'g-',label = 'Predicted Class Lables"Test')
# plt.title('Test Prediction')
# plt.legend(loc='center right')
# plt.savefig('./areds/saved_plots/areds_v3_predict.png')
#
# # Create 2-D array of True class and predicted class and save
# tst_stat = np.vstack((tst_lbls, test_pred)).T
# np.savetxt('./areds/saved_plots/areds_v3_predict.txt', tst_stat)
# dif = tst_lbls-test_pred
# # count number of zeros( where true class natches predicted class)
# print('test_acc_again:' + str(np.count_nonzero(dif==0)/nTest))

