
import numpy as np
#number of clean files , that will be picked at random, and concatanated
seq_range = np.arange(3,9)
# Targer SNR after mixing speech with noise
SNR_range = np.arange(10,31)
# number of such noisy speech sequences to be generated
no_seq = 5
# path to clean speech files
pth_cln = '//ussjf-mcorp.synaptics-inc.local//IoT_MLCOLD//Datasets//GarbageData//English//LibriSpeech//train-other-500'
# path to noise files
pth_noise = "//ussjf-mcorp.synaptics-inc.local//IoT_MLCOLD//Datasets//NoiseData//Train//PINK"
# path to impulse response files
pth_ir = "//ussjf-mcorp.synaptics-inc.local//IoT_MLCOLD//Datasets//RecordedIR//ir3"
# path for result mix files
pth_sv = ".//mixed//"
# save list of clean files in pth_cln, for faster load next time
save_cln_file = 1
load_cln_file = 0
