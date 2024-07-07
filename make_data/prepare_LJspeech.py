# -*- coding: utf-8 -*-
###############################################################################
import glob
import numpy as np
import pickle
import soundfile as sf
from tqdm import tqdm
np.random.seed(0)

from make_data.audio_sample_config import sample_list


## Data #######################################################################
wav_dir   = 'make_data/LJSpeech-1.1/wavs'
wav_files = glob.glob(wav_dir + "/" + "*.wav")
Num_data  = len(wav_files)
print('Num_data:', str(Num_data))

sample_list = [wav_dir + "/" + fname for fname in sample_list]
wav_files_ = list(set(wav_files) - set(sample_list))
wav_files_ = list(np.random.permutation(wav_files_))


## Parametrs ##################################################################
fs = 22050
winlen = 1024
shift  = 256
lf = 24064

num_train = 12500
num_valid = 300
num_test = 300


## Splitting ##################################################################
train_fns = wav_files_[:num_train]
valid_fns = wav_files_[num_train:num_train+num_valid]
test_fns  = wav_files_[num_train+num_valid:]
test_fns.extend(sample_list)

## Training data ##############################################################
for fn in tqdm(train_fns, desc='Training set'):
    speech_tmp, _ = sf.read(fn)
    num_iter = int(np.floor(len(speech_tmp)/lf))
    name = fn.split('/')[-1].split('.')[0]
    
    for i in range(num_iter):
        speech_ttmp = speech_tmp[i*lf:(i+1)*lf]
        speech_ttmp = np.concatenate((np.zeros(winlen-shift,),speech_ttmp,
                                          np.zeros(winlen-shift,)), axis=0)
        speech_ttmp /= np.max(np.abs(speech_ttmp))
        np.save('data/train/'+name+'_'+str(num_iter)+'.npy', speech_ttmp.astype(np.float32))

                             
## Validation data ############################################################
for fn in tqdm(valid_fns, desc='Validation set'):

    speech_tmp, _ = sf.read(fn)
    num_iter = int(np.floor(len(speech_tmp)/lf))
    name = fn.split('/')[-1].split('.')[0]                    
    
    for i in range(num_iter):
        speech_ttmp = speech_tmp[i*lf:(i+1)*lf]
        speech_ttmp = np.concatenate((np.zeros(winlen-shift,),speech_ttmp,
                                          np.zeros(winlen-shift,)), axis=0)
        speech_ttmp /= np.max(np.abs(speech_ttmp))
        np.save('data/valid/'+name+'_'+str(num_iter)+'.npy', speech_ttmp.astype(np.float32))

                             
## Test data ##################################################################
for fn in tqdm(test_fns, desc='Test set'):
    speech_ttemp, _ = sf.read(fn)
    speech_ttemp /= np.max(np.abs(speech_ttemp))
    name = fn.split('/')[-1].split('.')[0]
    np.save('data/test/'+name+'.npy', speech_ttemp.astype(np.float32))