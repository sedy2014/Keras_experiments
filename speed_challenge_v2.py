from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras import backend as K
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint ,EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
K.backend.image_data_format=  'channels_last'

load_mod = 0


import tensorflow as tf
from tensorflow.keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


#read each frame into jpeg image ,cropt it and savd cropped data

def vid_to_img(bs_pth,pthin,N,frms,nm_str):
    # create video object
    cap = cv2.VideoCapture(pthin)
    # set the number of frames to match length corresponding to labels
    cap.set(cv2.CAP_PROP_FRAME_COUNT, N)
    for frm_id, item in enumerate(frms):
        # CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be captured next
        cap.set(cv2.CAP_PROP_POS_FRAMES, frm_id)
        # read next frame ( 480,640,3)
        success, frm_read = cap.read()
        # get image coordinates
        N1 = frm_read.shape[0]
        N2 = frm_read.shape[1]
        pery1 = 0.3 # 0.5 matches online
        pery2 = 0.7
        perx1  = 0.86
        perx2 = 0.86
        if success:
            # # crop the imag
            #y1,y2,x1,x2 = crop_img_coord(frm_read,0.8,N1,N2)
            y1, y2, x1, x2 = crop1_img_coord(frm_read, pery1,pery2,perx1,perx2, N1, N2)
            frm_read = frm_read[y1:y2, x1:x2, :3]

            # Do contrast limited adaptive equalization
            frm_read_eq = clahe(frm_read)

            # frm_read = frm_read[100:440, :-90]
            # create path for writing the images
            # saved_frames1 had no clahe images
            image_path = bs_pth +  nm_str +  '//' +  str(frm_id + 1) + '.png'
            print('current frame = ' + str(frm_id))
            #if frm_id == 56:
                # save image to IMG folder
            cv2.imwrite(image_path, frm_read_eq)

# crop image whose size is (N1 rows,N2 col) , beginning from centre and getting per pixels from there
def crop_img_coord(img,per,N1,N2):
    # Get image centre
    ymid = np.floor(N1 / 2)
    xmid = np.floor(N2 / 2)
    # These many pixels
    dely = np.floor(ymid * per)
    delx = np.floor(xmid * per)

    y1 = int(ymid - dely)
    y2 = int(ymid + dely)
    x1 = int(xmid - delx)
    x2 = int(xmid + delx)
    return y1,y2,x1,x2

def crop1_img_coord(img,pery1,pery2,perx1,perx2,N1,N2):
    # Get image centre
    ymid = np.floor(N1 / 2)
    xmid = np.floor(N2 / 2)
    # These many pixels
    dely1 = np.floor(ymid * pery1)
    dely2 = np.floor(ymid * pery2)
    delx1 = np.floor(xmid * perx1)
    delx2 = np.floor(xmid * perx2)

    y1 = int(ymid - dely1)
    y2 = int(ymid + dely2)
    x1 = int(xmid - delx1)
    x2 = int(xmid + delx2)
    return y1,y2,x1,x2

# function for contrast limited Adaptive equalization
def clahe(img):
    # convert to LAB
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # split channels
    l_ch, a_ch, b_ch = cv2.split(lab_image)
    # apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch_equalized = clahe.apply(l_ch)
    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((l_ch_equalized, a_ch, b_ch))
    # convert iamge from LAB color model back to RGB color model
    eq_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return eq_image

def creat_dict(a):
    lbl_dict =  dict()
    arr = np.array([])
    cnt = 0
    for i in range(1,np.size(a)):
        arr = np.append(arr, i)
        if i%2 == 0:
            cnt = cnt + 1
            lbl_dict.update({cnt:tuple(arr)})
            arr = np.array([])
            continue
        print(i)
    return lbl_dict

def nvidia_model():
    Nh = 66
    Nw = 220
    Nc = 3

    model = Sequential()
    # custom normalization
    model.add(layers.Lambda(lambda x: x / 127.5 - 1, input_shape=(Nh, Nw, Nc)))
    model.add(layers.Conv2D(24, (5, 5), input_shape=(Nh, Nw, Nc), kernel_initializer = 'he_normal',activation='relu',padding='valid',name='conv1'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(36, (5, 5), kernel_initializer = 'he_normal',activation='relu', padding='valid',name='conv2'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(48, (5, 5), kernel_initializer = 'he_normal',activation='relu', padding='valid',name='conv3'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), kernel_initializer = 'he_normal',activation='relu', padding='valid',name='conv4'))
    model.add(layers.MaxPooling2D(pool_size=(1, 1)))
    #model.add(Conv2D(64, (3, 3), kernel_initializer = 'he_normal',activation=None, padding='valid',name='conv5'))
    #model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', activation='relu', padding='valid', name='conv5'))
    #model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(layers.Flatten(name = 'flatten'))
    #model.add(Activation('relu'))
    model.add(layers.Dense(100, kernel_initializer='he_normal',activation='elu', name='fc1'))
    model.add(layers.Dense(50, kernel_initializer='he_normal', activation='elu',name='fc2'))
    model.add(layers.Dense(10, kernel_initializer='he_normal', activation='elu',name='fc3'))

    # do not put activation at the end because we want to exact output, not a class identifier
    model.add(layers.Dense(1, name='output', kernel_initializer='he_normal' , activation=None))

   # adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #model.compile(optimizer=adam, loss='mse')

    return model


# def nvidia_model():
#     N_img_height = 66
#     N_img_width = 220
#     N_img_channels = 3
#     inputShape = (N_img_height, N_img_width, N_img_channels)
#
#     model = Sequential()
#
#     # perform custom normalization before lambda layer in network
#     model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=inputShape))
#
#     model.add(Convolution2D(24, (5, 5),
#                             strides=(2, 2),
#                             padding='valid',
#                             kernel_initializer='he_normal',
#                             name='conv1'))
#
#     model.add(ELU())
#     model.add(Convolution2D(36, (5, 5),
#                             strides=(2, 2),
#                             padding='valid',
#                             kernel_initializer='he_normal',
#                             name='conv2'))
#
#     model.add(ELU())
#     model.add(Convolution2D(48, (5, 5),
#                             strides=(2, 2),
#                             padding='valid',
#                             kernel_initializer='he_normal',
#                             name='conv3'))
#     model.add(ELU())
#     model.add(Dropout(0.5))
#     model.add(Convolution2D(64, (3, 3),
#                             strides=(1, 1),
#                             padding='valid',
#                             kernel_initializer='he_normal',
#                             name='conv4'))
#
#     model.add(ELU())
#     model.add(Convolution2D(64, (3, 3),
#                             strides=(1, 1),
#                             padding='valid',
#                             kernel_initializer='he_normal',
#                             name='conv5'))
#
#     model.add(Flatten(name='flatten'))
#     model.add(ELU())
#     model.add(Dense(100, kernel_initializer='he_normal', name='fc1'))
#     model.add(ELU())
#     model.add(Dense(50, kernel_initializer='he_normal', name='fc2'))
#     model.add(ELU())
#     model.add(Dense(10, kernel_initializer='he_normal', name='fc3'))
#     model.add(ELU())
#     model.add(Dense(1, name='output', kernel_initializer='he_normal'))
#     #adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#     #model.compile(optimizer=adam, loss='mse')
#     return model

def opticalFlowDense(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)

    hsv = np.zeros((66, 220, 3))
    # set saturation
    hsv[:, :, 1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:, :, 1]

    # Flow Parameters
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,
                                        flow_mat,
                                        image_scale,
                                        nb_images,
                                        win_size,
                                        nb_iterations,
                                        deg_expansion,
                                        STD,
                                        0)

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hue corresponds to direction
    hsv[:, :, 0] = ang * (180 / np.pi / 2)
    # value corresponds to magnitude
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # convert HSV to float32's
    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_flow

def ftr_gen(frms_dual,lbl,load_pth,frm_dict):
    sz = np.size(frms_dual)
    ftr = np.zeros((sz, 66, 220, 3))  # nvidia input params
    label = np.zeros((sz))
    cnt =0
    for i,v in enumerate(frms_dual):
        # read dictionary values
        [fr_cu,frm_nxt] = frm_dict[v]
        # load these 2 images
        src_pth1 = load_pth + '\\' + str(int(fr_cu)) + '.png'
        src_pth2 = load_pth + '\\' + str(int(frm_nxt)) + '.png'
        cnt = cnt +  1
        print('cnt = ' + str(cnt))
        #print(src_pth1)
        #print(src_pth2)
        img1 = cv2.imread(src_pth1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(src_pth2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # resize the image to expected size
        img1_rz = cv2.resize(img1, (220, 66), interpolation=cv2.INTER_AREA)
        img2_rz = cv2.resize(img2, (220, 66), interpolation=cv2.INTER_AREA)
        # compute optical flow on these 2  images as RGB
        oflow_img = opticalFlowDense(img1_rz, img2_rz)
        #average the label/speed
        y1 = lbl[int(fr_cu) - 1]
        y2 = lbl[int(frm_nxt) - 1]
        y = np.mean([y1, y2])
        ftr[i] = oflow_img
        label[i] = y
    return ftr,label

# if training the model, plot training and validation loss
def smooth_curve(data, alpha):
    smooth_d = []
    for point in data:
        if smooth_d:
            prev = smooth_d[-1]
            smooth_d.append((prev * alpha) + point * (1 - alpha))
        else:
            smooth_d.append(point)

    return smooth_d


def test_load_create(frms_dual,load_pth,frm_dict):
    # compute optical flow now
    sz = np.size(frms_dual)
    # nvidia input shape
    ftr = np.zeros((sz, 66, 220, 3))
    cnt = 0
    for i, v in enumerate(frms_dual):
        # read dictionary values
        [fr_cu, frm_nxt] = frm_dict[v]
        # # load these 2 images
        src_pth1 = load_pth + '\\' + str(int(fr_cu)) + '.png'
        src_pth2 = load_pth +  '\\' + str(int(frm_nxt)) + '.png'
        cnt = cnt + 1
        print('cnt = ' + str(cnt))
        # print(src_pth1)
        # print(src_pth2)
        img1 = cv2.imread(src_pth1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(src_pth2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # resize the image to expected size
        img1_rz = cv2.resize(img1, (220, 66), interpolation=cv2.INTER_AREA)
        img2_rz = cv2.resize(img2, (220, 66), interpolation=cv2.INTER_AREA)
        # compute optical flow on these 2  images as RGB
        oflow_img = opticalFlowDense(img1_rz, img2_rz)
        ftr[i] = oflow_img
    return ftr


#********************* main *************************
print (cv2.__version__)
bs_pth = './speed_challenge_2017/data'
train_pth = bs_pth + '/train.mp4'
test_pth = bs_pth + '/test.mp4'
train_lbl_pth = bs_pth + '/train.txt'
load_pth = bs_pth +  '/saved_frames'
load_pth_tst =  bs_pth +  '/saved_frames_test'


# load labels
all_lbl = np.loadtxt(train_lbl_pth)
N_fram = all_lbl.size
frms = np.arange(1,N_fram+1)
##print('No of frames = ' + str(N_fram))

load_tr_vid = 0
load_tst_vid = 0

# load video, read each frame, crop it, Histogram equalize it, and save each image
if load_tr_vid :
    vid_to_img(bs_pth,train_pth,N_fram,frms,'//saved_frames')

# create dictionary of frame ids's 2 at a time
frm_indx1 = np.append(frms,[N_fram + 1, N_fram + 2])
frm_dict= creat_dict(frm_indx1)
for key,val in frm_dict.items():
    print("{} = {}".format(key, val))
# compute half indices corresponding to the keys
frms_dual = np.arange(1,int(N_fram/2)+ 1)

# load the data saved last time
ld_dat = 1
if not(ld_dat):
    tr_frms_dual, val_frms_dual = train_test_split(frms_dual, train_size=0.8, test_size=0.2, shuffle=True)
    # create Training features array
    x_tr, y_tr = ftr_gen(tr_frms_dual, all_lbl, load_pth, frm_dict)
    # create validation features array
    x_val, y_val = ftr_gen(val_frms_dual, all_lbl, load_pth, frm_dict)

    h5f = h5py.File((bs_pth + '\\tr_frms_dual.h5'), 'w')
    h5f.create_dataset('tr_frms_dual', data=tr_frms_dual)
    h5f.close()
    h5f = h5py.File((bs_pth + '\\val_frms_dual.h5'), 'w')
    h5f.create_dataset('val_frms_dual', data=val_frms_dual)
    h5f.close()

    h5f = h5py.File((bs_pth + '\\x_tr.h5'), 'w')
    h5f.create_dataset('x_tr', data=x_tr)
    h5f.close()
    h5f = h5py.File((bs_pth + '\\y_tr.h5'), 'w')
    h5f.create_dataset('y_tr', data=y_tr)
    h5f.close()

    h5f = h5py.File((bs_pth + '\\x_val.h5'), 'w')
    h5f.create_dataset('x_val', data=x_val)
    h5f.close()
    h5f = h5py.File((bs_pth + '\\y_val.h5'), 'w')
    h5f.create_dataset('y_val', data=y_val)
    h5f.close()
else:
    h5f = h5py.File(bs_pth +'\\tr_frms_dual.h5', 'r')
    tr_frms_dual = h5f['tr_frms_dual'][:]
    h5f.close()
    h5f = h5py.File(bs_pth +'\\val_frms_dual.h5', 'r')
    val_frms_dual = h5f['val_frms_dual'][:]
    h5f.close()

    h5f = h5py.File(bs_pth +'\\x_tr.h5', 'r')
    x_tr = h5f['x_tr'][:]
    h5f.close()
    h5f = h5py.File(bs_pth +'\\y_tr.h5', 'r')
    y_tr = h5f['y_tr'][:]
    h5f.close()

    h5f = h5py.File(bs_pth +'\\x_val.h5', 'r')
    x_val = h5f['x_val'][:]
    h5f.close()
    h5f = h5py.File(bs_pth +'\\y_val.h5', 'r')
    y_val = h5f['y_val'][:]
    h5f.close()

# load previous model or train again
load_mod = 0
fit_model = 1
if load_mod:
     json_file = open(bs_pth + "//save_model_speed_challenge.json", 'r')
     loaded_model_json = json_file.read()
     json_file.close()
     model = model_from_json(loaded_model_json)
     #model.load_weights(bs_pth + "//speed_challenge_weights.h5")
     model.load_weights(bs_pth + "//speed_challenge_best.h5")
     model.summary()
else:
    # instantiate the model and compile
    model = nvidia_model()
    model.summary()

if fit_model:
    batch_size = 32
    model.compile(optimizer=optimizers.Adam(lr=1e-3),loss='mse')
    #model.compile(optimizer=optimizers.RMSprop(lr=1e-3), loss='mse')
    # monitoring validation accuracy, and saving only the weights  for best epoch
    checkpointer = ModelCheckpoint(filepath=bs_pth + "//speed_challenge_best.h5",
                                   monitor = 'val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.8,patience = 5,min_lr= 1e-5,verbose=1)
    # show what happened in checkpoint
    callbacks_list = [checkpointer ,reduce_lr]
    steps_per_epoch_tr = int(np.size(tr_frms_dual)/batch_size)
    steps_per_epoch_val = int(np.size(val_frms_dual)/batch_size)
    num_epoch =  75

    mdlfit = model.fit(x_tr,y_tr, epochs=num_epoch,batch_size=batch_size,validation_data=(x_val,y_val),callbacks=callbacks_list)

    model.save_weights(bs_pth + '//speed_challenge_weights.h5')
    model_json = model.to_json()
    with open(bs_pth + '//save_model_speed_challenge.json', "w") as json_file:
        json_file.write(model_json)
        json_file.close()

    preds = model.predict(x_val, verbose=1)
    print( "Val MSE: {}".format(mean_squared_error(y_val, preds)))

    #plot training and validtion loss
    trn_loss = mdlfit.history['loss']
    val_loss = mdlfit.history['val_loss']
    epochs = range(len(trn_loss))
    plt.plot(epochs,trn_loss,'b-',label = 'Train Loss')
    plt.plot(epochs,val_loss,'g-',label = 'Valid Loss')
    plt.title('Loss')
    plt.legend(loc='center right')
    plt.savefig( bs_pth + '//loss.png')
    plt.show()
    plt.plot(epochs,smooth_curve(trn_loss,0.8),'b-',label = 'Smoothed Train Loss')
    plt.plot(epochs,smooth_curve(val_loss,0.8),'g-',label = 'Smoothed Valid Loss')
    plt.title('Smoothed Loss')
    plt.legend(loc='center right')
    plt.savefig( bs_pth + '//smooth_loss.png')

    h5f = h5py.File((bs_pth + '\\tr_loss.h5'), 'w')
    h5f.create_dataset('tr_loss', data=trn_loss)
    h5f.close()
    h5f = h5py.File((bs_pth + '\\val_loss.h5'), 'w')
    h5f.create_dataset('val_loss', data=val_loss)
    h5f.close()
else:
    print("")
    h5f = h5py.File(bs_pth + '\\tr_loss.h5', 'r')
    trn_loss = h5f['tr_loss'][:]
    h5f.close()
    h5f = h5py.File(bs_pth + '\\val_loss.h5', 'r')
    val_loss = h5f['val_loss'][:]
    h5f.close()
    epochs = range(len(trn_loss))
    plt.plot(epochs, trn_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'g-', label='Valid Loss')
    plt.title('Loss')
    plt.legend(loc='center right')
    plt.show()
    plt.plot(epochs, smooth_curve(trn_loss, 0.8), 'b-', label='Smoothed Train Loss')
    plt.plot(epochs, smooth_curve(val_loss, 0.8), 'g-', label='Smoothed Valid Loss')
    plt.title('Smoothed Loss')
    plt.legend(loc='center right')


# predict on test data
cap = cv2.VideoCapture(test_pth)
N_frames_test = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frms_tst = np.arange(1,N_frames_test+1)

# create dictionary of frame ids's 2 at a time
frm_indx1_tst = np.append(frms_tst,[N_frames_test + 1, N_frames_test + 2])
frm_dict_tst= creat_dict(frm_indx1_tst)
for key,val in frm_dict_tst.items():
    print("{} = {}".format(key, val))

# compute half indices corresponding to the keys
frms_dual_tst = np.arange(1,int(N_frames_test/2)+ 1)
#save frames
if load_tst_vid :
    vid_to_img(bs_pth,test_pth,N_frames_test,frms_tst,'//saved_frames_test')

# load saved frames and convert to features
'//saved_frames_test'
x_tst = test_load_create(frms_dual_tst,load_pth_tst,frm_dict_tst)
y_pre = model.predict(x_tst,verbose=1)
np.savetxt(bs_pth + '\\test_pred.txt',y_pre,fmt='%f')