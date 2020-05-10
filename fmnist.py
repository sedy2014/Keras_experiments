# wrt v4,modified code to add functions instead of all the code here


from __future__ import print_function
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras import backend as k
k.backend.image_data_format=  'channels_last'  # for remote run
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers,models
from keras.models import  model_from_json
from tensorflow.keras.applications import InceptionResNetV2,MobileNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
#import cv2
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import multi_gpu_model
from helper_func import save_batch_info,smooth_curve,save_model_hyperparam,save_plots,get_batch_stats
import tensorflow as tf
import os
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

# If loading saved weights, set to 1, else 0
load_mod = 0
save_model_weights = 1
save_best_weights = True # if 0, saves weights after last epoch
multi_gpu = 0
save_tr_val_stat_plots = 1 #save accuracy,loss plots for tr and val data
# run on subset of data
run_subset = 0
# model options : 1 (resnet) 2: MobilenetV2   3: self
model_opt = 2
if not load_mod:
    if model_opt == 1:
        # number of layers to freeze for not training
        FREEZE_LAYERS = 2
    elif model_opt == 2:
        FREEZE_LAYERS = 0
    else:
        FREEZE_LAYERS = 0

# save flenames used in each batch of training nd validation data
save_batch_stats = 1

# get current filename without extension
fnm = os.path.basename(__file__)
fnm    = os.path.splitext(fnm)[0]
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")

bs_pth = './/color_mnist//synthetic_digits'

if  run_subset == 0:
    train_dir = bs_pth + '//imgs_train'
    validation_dir = bs_pth + '//imgs_valid'
    test_dir = bs_pth + '//imgs_test'
else:
    train_dir = bs_pth + '//input_subset//imgs_train'
    validation_dir = bs_pth + '//input_subset//imgs__valid'
    test_dir = bs_pth + '//input_subset//imgs_test'

output = bs_pth + '//output'
pth_saved_model = bs_pth + '/saved_model' + '/' + dt_string
if not load_mod and not os.path.exists(pth_saved_model):
    os.makedirs(pth_saved_model)
pth_saved_model_json =  pth_saved_model +  '/' + fnm + '.json'
pth_saved_model_weights =  pth_saved_model +  '/' + fnm + '_weights.h5'

pth_saved_plots = bs_pth + '/saved_plots' +  '/' + dt_string
if not os.path.exists(pth_saved_plots):
    os.makedirs(pth_saved_plots)

if save_batch_stats :
    pth_batch_tr = pth_saved_plots + '/' +  fnm + '_batch_tr.txt'
    pth_batch_vd = pth_saved_plots + '/' +  fnm + '_batch_vd.txt'

tr_per = 0.8
val_per = 0.2
tst_per = 1 - tr_per - val_per
mv_cpy=0
shuffle = 'True'
class_nms = ['0','1','2','3','4','5','6','7','8','9']

W = 299
H = 299
nc = 3
num_pixels = W*H
nclass = len(class_nms)
resz_img = 1
# if running locally, resize to smaller size to be able to use larger batch size
if resz_img:
    W_rz = 96
    H_rz = 96
    num_pixels = W_rz * H_rz
else:
    W_rz = W
    H_rz = H
if load_mod:
    # This is really slow, so avoid it
    #model = load_model('./areds/saved_model/areds_v5.h5') C:\ML\env\tf\areds\saved_model
    json_file = open('C://ML//env//tf//glucoma//saved_model//06_02_2020_17_11//areds_v5.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #Load weights saved from last epoch or the best epoch
    model.load_weights('C://ML//env//tf//glucoma//saved_model//06_02_2020_17_11//areds_v5_bestWeights.h5')
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
    def baseline_model():
        # create model
        model = models.Sequential()
        # flatten input to (N1,w*h) as fit_generator expects (N1,w*h), but dont' have x,y as inputs(so cant reshape)
        model.add(layers.Flatten(input_shape=(H_rz, W_rz, nc)))
        model.add(layers.Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
        # Define output layer with softmax function
        model.add(layers.Dense(nclass, kernel_initializer='normal', activation='softmax'))
        return model
    # mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224
    def Mobilenetv2_finetune_model():
        MobileNet_tuned = MobileNetV2(weights='C:\\ML\env\\tf\\pycharm_areds3\\areds\\saved_model\\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5'
                                              , include_top=False,input_shape=(W_rz, H_rz, nc),alpha =1)
        #If alpha < 1.0, proportionally decreases the number of filters in each layer.
        # If alpha > 1.0, proportionally increases the number of filters in each layer.
        model = models.Sequential()
        # add Resnet as the base( no need to specify input size here)
        model.add(MobileNet_tuned)
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(nclass, activation='softmax'))
        return model
    if model_opt == 1:
        model = Resnet_finetune_model()
        # instantiate the Resnet base model
        conv_bs = model.get_layer('inception_resnet_v2')
        print("This is no of trainable weights in base Resnet " + str(len(conv_bs.trainable_weights)))
        print("This is no of trainable weights in base Resnet and added dense layers before freezing" + str(
            len(model.trainable_weights)))
        conv_bs.trainable = False
        # freeze the first this many layers for training\
        for layer in conv_bs.layers[:FREEZE_LAYERS]:
            layer.trainable = False
        for layer in conv_bs.layers[FREEZE_LAYERS:]:
            layer.trainable = True
        print("This is no of trainable weights in base Resnet and added dense layers after freezing" + str(
            len(model.trainable_weights)))
        for layer in conv_bs.layers:
            print(layer.name + "Trainable staus: " + str(layer.trainable) + "\n")
    elif model_opt == 2:
        model = Mobilenetv2_finetune_model()
        # instantiate the Mobilenetv2 base model
        conv_bs = model.get_layer('mobilenetv2_1.00_96')
        print("This is no of trainable weights in base MobilenetV2 " + str(len(conv_bs.trainable_weights)))
        print("This is no of trainable weights in base MobilenetV2 and added dense layers before freezing" + str(
            len(model.trainable_weights)))
        conv_bs.trainable = True
        #freeze the first this many layers for training\
        # for layer in conv_bs.layers[:FREEZE_LAYERS]:
        #     layer.trainable = False
        # for layer in conv_bs.layers[FREEZE_LAYERS:]:
        #     layer.trainable = True
        print("This is no of trainable weights in base MobilenetV2 and added dense layers after freezing" + str(
            len(model.trainable_weights)))
        for layer in conv_bs.layers:
            print(layer.name + "Trainable staus: " + str(layer.trainable) + "\n")
    elif model_opt == 3:
        model = baseline_model()
    else:
        print("no model specified")
    model.summary()

# if not loadin model, train it
if not load_mod :
    # print learning rates
    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr

        return lr
    batch_size = 156
    epochs = 25
    lr_st = 0.001/2
    optimizer = optimizers.Adam(lr=lr_st)
    lr_metric = get_lr_metric(optimizer)
    loss_func = 'categorical_crossentropy'
    # datagen_tr  = ImageDataGenerator(preprocessing_function=preprocess_input,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,channel_shift_range=10,horizontal_flip=True,fill_mode='nearest')
    datagen_tr = ImageDataGenerator(preprocessing_function=preprocess_input)
    datagen_vd = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_gen = datagen_tr.flow_from_directory(train_dir, target_size=(W_rz, H_rz), batch_size=batch_size,
                                               class_mode='categorical', shuffle=True, interpolation="bilinear")
    vd_gen = datagen_vd.flow_from_directory(validation_dir, target_size=(W_rz, H_rz), batch_size=batch_size,
                                            class_mode='categorical', shuffle=False, interpolation="bilinear")
    nTrain = np.size(train_gen.classes)
    nVal = np.size(vd_gen.classes)
    steps_per_epoch_tr = int(nTrain / batch_size)
    steps_per_epoch_val = int(nVal / batch_size)
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(train_gen.classes),train_gen.classes)
    # ***********************  print batch  data to files *************************
    #if save_batch_stats:
       # get_batch_stats(train_gen, pth_batch_tr)
        # save_batch_info(train_gen, pth_batch_tr)
        # save_batch_info(vd_gen, pth_batch_vd)
    if multi_gpu :
        parallel_model = multi_gpu_model(model, gpus=4)
        parallel_model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy', lr_metric])
        reduce_lr_cbk = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1,min_delta=1e-3, cooldown=0)
        # monitoring validation accuracy, and saving only the weights  for best epoch
        checkpointer = ModelCheckpoint(filepath=pth_saved_model_weights, monitor='val_acc',verbose=1, save_best_only=True)
       # tb_cbk = TensorBoard(log_dir=pth_saved_model, histogram_freq=0, write_graph=True, write_images=True)
        # show what happened in checkpoint
        #callbacks_list = [reduce_lr_cbk, checkpointer]
        callbacks_list = [checkpointer]
        # callbacks_list = [reduce_lr_cbk, checkpointer,tb_cbk]
        if len(callbacks_list) == 0:
            callbacks_list = []
        # This `fit` call will be distributed on 8 GPUs.
        # Since the batch size is 256, each GPU will process 32 samples.
        parallel_model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = vd_gen,
                    validation_steps = steps_per_epoch_val,epochs=epochs,callbacks=callbacks_list,verbose=1)
    else:
        model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy', lr_metric])
        reduce_lr_cbk = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,  verbose=1, min_delta=1e-3,cooldown=1)
        # monitoring validation accuracy, and saving only the weights  for best epoch
        checkpointer = ModelCheckpoint(pth_saved_model_weights,monitor = 'val_acc',verbose = 1,save_best_only = True)
        #tb_cbk = TensorBoard(log_dir= pth_saved_model, histogram_freq=0, write_graph=True, write_images=True)
        # show what happened in checkpoint
        #callbacks_list = [reduce_lr_cbk,checkpointer]
        callbacks_list = [checkpointer]
        #callbacks_list = [reduce_lr_cbk, checkpointer,tb_cbk]
        if len(callbacks_list) == 0:
            callbacks_list = []
        # mdlfit=model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = vd_gen,
        #                     validation_steps = steps_per_epoch_val,epochs=epochs,class_weight=class_weights,callbacks=callbacks_list)
        mdlfit=model.fit_generator(train_gen,steps_per_epoch = steps_per_epoch_tr,validation_data = vd_gen,
                validation_steps = steps_per_epoch_val,epochs=epochs,callbacks=callbacks_list,verbose=1)
    if save_model_weights:
        # write hyperparams to text file
        save_model_hyperparam(model, pth_saved_model, pth_batch_tr, pth_batch_vd, load_mod, save_model_weights,
                              save_best_weights,train_gen, vd_gen, batch_size, epochs, lr_st, optimizer, loss_func,
                              callbacks_list)
        # model.save_weights(pth_saved_model + '/areds_v5_weights.h5')
        # model_json = model.to_json()
        # with open(pth_saved_model + '/areds_v5.json', "w") as json_file:
        #     json_file.write(model_json)
        #     json_file.close()
# if training the model, plot training and validation accuracy
if not load_mod :
    if save_tr_val_stat_plots:
        save_plots(mdlfit, pth_saved_plots, fnm)
