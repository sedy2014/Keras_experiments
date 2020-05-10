# from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from helper_func import tr_va_ts_split, create_subset, get_class_stats,get_batch_stats,save_batch_info
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import os
from datetime import datetime
bs_pth = 'E://Data//55753//AREDS//jpeg_test'
train_dir = bs_pth + '//train'
validation_dir = bs_pth + '//valid'
test_dir = bs_pth + '//test'
tr_per = 0.8
val_per = 0.2
tst_per = 0
#tst_per = 1 - tr_per - val_per
mv_cpy=0
shuffle = 'True'
class_nms = ['control','NV','GA']

# x= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,18,19,20]
# x= np.asarray(x)
# x= np.arange(0,10)
# df = pd.DataFrame(x)
# #np.random.seed(seed)
# perm = np.random.permutation(df.index)
# m = len(df.index)
# train_end = int(tr_per * m)
# validate_end = int(val_per * m) + train_end
# train = df.ix[perm[:train_end]]
# validate = df.ix[perm[train_end:validate_end]]
# test = df.ix[perm[validate_end:]]
#
# train= train.to_numpy()
# validate =  validate.to_numpy()
# test =  test.to_numpy()

#all_files,tr_indx,val_indx,tst_indx,dest_dir_tr,dest_dir_val,dest_dir_tst = tr_va_ts_split(bs_pth,tr_per,val_per,shuffle,class_nms,mv_cpy)

# create sunset data
#create_subset('C:/ML//env//tf//pycharm_areds3//areds//input',0.04)
#subfolder_names, class_names, class_samples ,class_samples_norm = get_class_stats('C:/ML//env//tf//pycharm_areds3//areds//input')
#create_subset('C://ML//env//tf//glucoma//no_resize', 0.3)
subfolder_names, class_names, class_samples ,class_samples_norm = get_class_stats('C://ML//env//tf//glucoma//no_resize_subset')

# get class stats
W_rz = 224
H_rz = 224
batch_size = 32

run_subset = 1
save_batch_stats = 1
# get current filename without extension
fnm = os.path.basename(__file__)
fnm    = os.path.splitext(fnm)[0]
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M")
bs_pth = './/pycharm_areds3//areds'

if run_subset == 1:
    train_dir = bs_pth + '//input_subset//train'
    validation_dir = bs_pth + '//input_subset//valid'
    test_dir = bs_pth + '//input_subset//test'
else:
    train_dir = bs_pth + '//input//train'
    validation_dir = bs_pth + '//input//valid'
    test_dir = bs_pth + '//input//test'
pth_saved_plots = bs_pth + '/saved_plots' + '/' + dt_string
if not os.path.exists(pth_saved_plots):
    os.makedirs(pth_saved_plots)

if save_batch_stats :
    pth_batch_tr = pth_saved_plots + '/' +  fnm + '_batch_tr.txt'
    pth_batch_vd = pth_saved_plots + '/' +  fnm + '_batch_vd.txt'
subfolder_names, class_names, class_samples ,class_samples_norm = get_class_stats('C:/ML//env//tf//pycharm_areds3//areds//input')

datagen_tr = ImageDataGenerator(preprocessing_function=preprocess_input)
datagen_vd = ImageDataGenerator(preprocessing_function=preprocess_input)
# train_gen = datagen_tr.flow_from_directory(train_dir, target_size=(W_rz, H_rz), batch_size=batch_size,
#                                            class_mode='categorical', shuffle=True, interpolation="bilinear")
# vd_gen = datagen_vd.flow_from_directory(validation_dir, target_size=(W_rz, H_rz), batch_size=batch_size,
#                                         class_mode='categorical', shuffle=False, interpolation="bilinear")


# save_batch_info(train_gen, pth_batch_tr)
# get_batch_stats(train_gen, pth_saved_plots)
# save_batch_info(vd_gen, pth_batch_vd)
# get_batch_stats(vd_gen, pth_saved_plots)
# check resize images
vd_gen = datagen_vd.flow_from_directory(validation_dir, target_size=(W_rz, H_rz), batch_size=batch_size,save_to_dir= pth_saved_plots,
                                        class_mode='categorical', shuffle=False, interpolation="bilinear")
for i in range(100):
    vd_gen.next()
print('hi')

