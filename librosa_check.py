import os
import glob
import soundfile as sf
import random
import librosa
import numpy as np
import  scipy.io  as sio
from scipy import signal


x = np.arange(0,9).T
y =  np.arange(1,10).T
z= np.vstack((x,y)).T
x_rms = librosa.feature.rmse(y=x.astype(float), frame_length=3,hop_length=3, center=False, pad_mode='constant')
y_rms = librosa.feature.rmse(y=y.astype(float), frame_length=3,hop_length=3, center=False, pad_mode='constant')
z_rms_sum = (x_rms + y_rms)/2
z_rms = librosa.feature.rmse(y=z.T.astype(float), frame_length=3,hop_length=3, center=False, pad_mode='constant')
print('')