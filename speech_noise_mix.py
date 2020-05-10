import os
import glob
import soundfile as sf
import random
import librosa
import numpy as np
import  scipy.io  as sio
from scipy import signal
# this code will create noisy reverberated data

pth_config = ".\\config.py"
from config import seq_range,SNR_range,pth_cln,pth_noise,pth_ir,pth_sv,save_cln_file,load_cln_file,no_seq

#************** speech file list  ***********************
if load_cln_file :
    dat = sio.loadmat('.//clean_files.mat')
    pth_cln_aud = dat.get('arr').tolist()
else:
    # read all the file names in list
    dirs = os.listdir( pth_cln )
    pth_cln_aud = []
    #pth_cln_aud_arr = np.zeros(0)
    pth_cln_aud_arr = np.asarray(pth_cln_aud)
    for path, subdirs_categ, files in os.walk(pth_cln):
        if len(files) != 0:
            a = glob.glob(path + "//*.flac")
            pth_cln_aud = pth_cln_aud + a
            #np.append(pth_cln_aud_arr,np.asarray(a))
    if save_cln_file:
        sio.savemat('.//clean_files.mat' , mdict={'arr': pth_cln_aud})

# load the Impulse response folder
ir_files = glob.glob(pth_ir + "//*.wav")

# read all the noise file names in a list
dirs = os.listdir(pth_noise)
pth_ns_aud = []
for path1, subdirs1_categ, files1 in os.walk(pth_noise):
    if len(files1) != 0:
        b = glob.glob(path1 + "//*.wav")
        pth_ns_aud = pth_ns_aud + b


#****************************************************************

for i in range(no_seq):
    #******** Speech sequence generation ******
    # choose random number of speech files to concatanate into one sequence
    num_cln_read = random.choices(seq_range, k=1)
    # choose  some clean files randomly
    cln_speech_rand = random.choices(pth_cln_aud, k=num_cln_read[0])
    cln_speech_rand_merg = np.array([])
    # merge the files into a single file
    for f in cln_speech_rand:
        data, fs1 = sf.read(f, dtype='float32')
        cln_speech_rand_merg = np.append(cln_speech_rand_merg,data)

    # choose 2 channels belonging to the same room for speech and noise data
    while True:
        ir_rand = random.choices(ir_files, k=4)
        ir1_nm = os.path.basename(ir_rand[0])
        ir2_nm = os.path.basename(ir_rand[1])
        ir3_nm = os.path.basename(ir_rand[2])
        ir4_nm = os.path.basename(ir_rand[3])
        if  len(ir1_nm.split('_')) > 1:
            nm1 = ir1_nm.split('_')[0] + ir1_nm.split('_')[1]
        else:
            nm1 = ir1_nm.split('_')[0]
        if  len(ir2_nm.split('_')) > 1:
            nm2 = ir2_nm.split('_')[0] + ir2_nm.split('_')[1]
        else:
            nm2 = ir2_nm.split('_')[0]
        if  len(ir3_nm.split('_')) > 1:
            nm3 = ir3_nm.split('_')[0] + ir3_nm.split('_')[1]
        else:
            nm3 = ir3_nm.split('_')[0]
        if  len(ir4_nm.split('_')) > 1:
            nm4 = ir4_nm.split('_')[0] + ir4_nm.split('_')[1]
        else:
            nm4 = ir4_nm.split('_')[0]
        # assuming first two string literals separated by underscore define a room
        # break out of loop when these match correspoing to same room
        if ((nm1 == nm2) and (nm2 ==nm3) and (nm3 == nm4)) :
            break
     # read the chosen impulse response files
    ir1,fs = sf.read(ir_rand[0],dtype='float32')
    ir2,fs =  sf.read(ir_rand[1],dtype='float32')

    # create stereo  speech reverberated file
    a1 = signal.fftconvolve(cln_speech_rand_merg,ir1,mode='same')  # full: N1+N2 -1, same:N1
    # normalize
    a1 = a1/np.max(np.abs(a1))
    a2= signal.fftconvolve(cln_speech_rand_merg,ir2,mode='same')
    a2 = a2/np.max(np.abs(a2))
    cln_speech_rand_merg_rev=np.vstack((a1,a2)).T

    #********* noise data*******************

    # choose  a noise file randomly
    ns_file = random.choices(pth_ns_aud, k=1)
    # read the noise file
    ns_rand, fs2 = sf.read(ns_file[0], dtype='float32')
    # choose ir files from same room
    ir3,fs = sf.read(ir_rand[2],dtype='float32')
    ir4,fs =  sf.read(ir_rand[3],dtype='float32')

    # check that speech and noise and IR data is of same sample rate, else resample
    if fs1 != fs2:
        librosa.resample(ns_rand,fs2,fs1,fix='True')
    if fs1 != fs:
        librosa.resample(ir1, fs2, fs, fix='True')
        librosa.resample(ir2, fs2, fs, fix='True')
        librosa.resample(ir3, fs2, fs, fix='True')
        librosa.resample(ir4, fs2, fs, fix='True')

    # create stereo  noise reverberated file
    a3 = signal.fftconvolve(ns_rand,ir3,mode='same')  # full: N1+N2 -1, same:N1
    # normalize
    a3 = a3/np.max(np.abs(a3))
    a4= signal.fftconvolve(ns_rand,ir4,mode='same')
    a4 = a4/np.max(np.abs(a4))
    ns_rand_rev=np.vstack((a3,a4)).T

    # combine Speech  and noise as per SNR
    # check  the length of speech and Noise files
    L1 = cln_speech_rand_merg_rev.shape[0]
    L2 = ns_rand_rev.shape[0]
    # speech is longer
    if L1 > L2 :
        # Copy noise file multiple times
        m = int(np.fix(L1/L2))
        x = np.tile(ns_rand_rev,[m,1])
        y = ns_rand_rev[0: L1 -(m*L2),:]
        ns_rand_rev = np.vstack((x,y))
    elif L1 < L2 :
        # extract noise file upto speech file
        # choose start index for noise file , minimum possible 0 and maximum L2-L1
        noise_start_idx = np.random.randint(low=0, high= L2 - L1)
        ns_rand_rev = ns_rand_rev[noise_start_idx:noise_start_idx + L1,:]

    # Compute root-mean-square (RMS) energy for each frame
    #librosa rmse computes rmse over each channel, then  averages the result over channels
    noise_MeanRMS = np.mean(librosa.feature.rmse(y=ns_rand_rev.T.astype(float), frame_length=128,hop_length=128, center=False, pad_mode='constant'))
    clean_seq_RMS = np.sort(librosa.feature.rmse(y=cln_speech_rand_merg_rev.T.astype(float), frame_length=256,hop_length=256, center=False, pad_mode='constant'))
    clean_seq_PeakRMS = clean_seq_RMS[0,np.round(clean_seq_RMS.shape[1]*0.97).astype(int)]
    # Compute SNR from clean speech and noisy speech
    origSnr_dB = 20 * np.log10(clean_seq_PeakRMS / noise_MeanRMS)
    # compute factor by which expected SNR is different from waveform SNR
    rand_SNR_dB = random.choices(SNR_range, k=1)
    noiseScale_dB = origSnr_dB - rand_SNR_dB
    alpha = 10 ** (noiseScale_dB / 20)
    # combine at expected SNR
    seq = cln_speech_rand_merg_rev + (alpha * ns_rand_rev)
    # write the final sequence
    if not os.path.exists(pth_sv):
        os.makedirs(pth_sv)
    sf.write(pth_sv + str(i)+'.wav',seq,fs1,'PCM_16')

print("hi")