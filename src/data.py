


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

def postprocess(x_out, len_x, batch=False):
    x_out = x_out.numpy()
    if batch:
        x_out = x_out[:,winlen-shift:winlen-shift+len_x]
    else:
        x_out = x_out[winlen-shift:winlen-shift+len_x]
    x_out = normalize_1d(x_out)
    return x_out


data_folder = 'data/'
test_files = [os.path.join(data_folder, 'test', name) for name in os.listdir(os.path.join(data_folder, 'test'))]

class AudioTestDataset(Dataset):
    def __init__(self, files):
        super(AudioTestDataset, self).__init__()
        self.files = files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        name = self.files[index]
        data = np.load(name)
        len_x = len(data)
        data = np.concatenate((np.zeros(winlen-shift,),
                                          data ,
                                          np.zeros(winlen-shift,)), axis=0)
        data = zero_pad(data, winlen, shift)
        x = torch.FloatTensor(data)
        X = stft(x)
        amplitude = torch.abs(X)
        return X, amplitude, len_x