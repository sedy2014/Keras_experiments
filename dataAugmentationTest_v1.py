
from __future__ import print_function
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img,array_to_img, img_to_array
from keras import backend as K
import scipy.io
from sklearn.preprocessing import StandardScaler
K.set_image_dim_ordering('tf')



W=512
H=512


ip_pth = './diabeticRetino/test_feature_scal/train'

train_dir_dest = './diabeticRetino/test_feature_scal/aug1_train'
batch_size = 2

# read all files per class
ip_pth_files = ip_pth +'/**/*.png'
filelist = glob.glob(ip_pth_files,recursive=True)

x = np.array([np.array(Image.open(fname)) for fname in filelist])

N1 = x.shape[0]  # training size
h = x.shape[1]
w = x.shape[2]
c = x.shape[3]
x_t = x.reshape(N1, w, h, c).astype('float64')

# scaler = StandardScaler()
# scaled_x_t = scaler.fit_transform(x_t)
# print(scaled_x_t.mean(axis = 0))
# print(scaled_x_t.std(axis = 0))
all_mean = x_t.mean()
all_std = x_t.std()
all_var = x_t.var()

# ****verify feature mean*****
# print mean of 1st image
img1_mn = x_t[0,:,:,:].mean()
img2_mn = x_t[1,:,:,:].mean()
img3_mn = x_t[2,:,:,:].mean()
img4_mn = x_t[3,:,:,:].mean()
all_mn = (img1_mn + img2_mn  +img3_mn + img4_mn)/4
print('Numpy mean = ' + str(all_mean) + "  Manual mean = " + str(all_mn))

#**** verify feature sd****

x_t_flat = x_t.reshape(N1*w*h*c,1)
al_var=0
for x_dat in x_t_flat:
    al_var = al_var + ((x_dat-all_mean)**2)
al_var = al_var/(x_t_flat.size)
al_std =  al_var**0.5
print('Numpy std = ' + str(all_std) + "  Manual std = " + str(al_std))

datagen_norm_ftr =  ImageDataGenerator(featurewise_center= True,featurewise_std_normalization=True)
datagen_norm_ftr.fit(x_t,augment=True)
d1 = datagen_norm_ftr.mean[0][0][0]
img_all_mn_dg = (datagen_norm_ftr.mean[0][0][0].astype('float64') + datagen_norm_ftr.mean[0][0][1].astype('float64') + datagen_norm_ftr.mean[0][0][2].astype('float64'))/3
# get mean as per channel


imgR_mn = x_t[:,:,:,0].mean().astype('float64')
imgG_mn = x_t[:,:,:,1].mean()
imgB_mn = x_t[:,:,:,2].mean()
img_all_mn = (imgR_mn + imgG_mn + imgB_mn)/3

print('ImageDataGen Red Channel mean = ' + str(datagen_norm_ftr.mean[0][0][0].astype('float64')) + "  Manual mean = " + str(imgR_mn))
print('ImageDataGen Green Channel mean = ' + str(datagen_norm_ftr.mean[0][0][1].astype('float64')) + "  Manual mean = " + str(imgG_mn))
print('ImageDataGen Blue Channel mean = ' + str(datagen_norm_ftr.mean[0][0][2].astype('float64')) + "  Manual mean = " + str(imgB_mn))
print('ImageDataGen All Channel mean = ' + str(img_all_mn_dg) + "  Manual mean = " + str(img_all_mn))
n=0
for x_batch in datagen_norm_ftr.flow(x_t, y=None,shuffle = False,batch_size=4):
    # verify 0 mean 1 unity std
    print('Normalized data mean:',x_batch.mean())
    print('Normalized data mstd:',x_batch.std())
    n = n + 1
    if n == 1:
        break
n=0
# save augmented images
for x_batch in datagen_norm_ftr.flow(x_t[0:2,:,:,:], y=None,save_to_dir = train_dir_dest+'/0',shuffle = False,batch_size=2):
    # verify 0 mean 1 unity std
    print('Normalized data mean:',x_batch.mean())
    print('Normalized data mstd:',x_batch.std())
    n = n + 1
    if n == 1:
        break
n=0
for x_batch in datagen_norm_ftr.flow(x_t[2:4,:,:,:], y=None,save_to_dir = train_dir_dest+'/1',shuffle = False,batch_size=2):
    # verify 0 mean 1 unity std
    print('Normalized data mean:',x_batch.mean())
    print('Normalized data mstd:',x_batch.std())
    n = n + 1
    if n == 1:
        break
