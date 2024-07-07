import numpy as np
import os
import torch
from torch.utils.data import Dataset


def zero_pad(x, win_len, shift):
    lf  = len(x)
    T   = int(np.ceil(lf-win_len)/shift) + 1
    lf2 = win_len + T*shift
    x   = np.concatenate((x,np.zeros(lf2+shift-lf,)), axis=0)
    return x

def normalize_1d(signal, maxval=(2.**15-1.)/2**15):
    max_data = np.max(np.abs(signal))
    signal   = signal / max_data * maxval
    return signal

def postprocess(x_out, len_x, winlen, shift, batch=False):
    x_out = x_out.numpy()
    if batch:
        x_out = x_out[:,winlen-shift:winlen-shift+len_x]
    else:
        x_out = x_out[winlen-shift:winlen-shift+len_x]
    x_out = normalize_1d(x_out)
    return x_out



class AudioDataset(Dataset):
    def __init__(self, files, stft):
        super(AudioDataset, self).__init__()
        self.files = files
        self.stft = stft

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        name = self.files[index]
        data = np.load(name)
        x = torch.FloatTensor(data)
        X = self.stft(x)
        amplitude = torch.abs(X)
        return X, amplitude


class AudioTestDataset(Dataset):
    def __init__(self, files, stft, winlen, shift):
        super(AudioTestDataset, self).__init__()
        self.files = files
        self.stft = stft
        self.winlen = winlen
        self.shift = shift

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        name = self.files[index]
        data = np.load(name)
        len_x = len(data)
        data = np.concatenate((np.zeros(self.winlen-self.shift,),
                                          data ,
                                          np.zeros(self.winlen-self.shift,)), axis=0)
        data = zero_pad(data, self.winlen, self.shift)
        x = torch.FloatTensor(data)
        X = self.stft(x)
        amplitude = torch.abs(X)
        return X, amplitude, len_x