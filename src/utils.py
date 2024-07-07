


# STFT/iSTFT
class STFT():
    def __init__(self, winlen, shift, nfft):
        self.winlen = winlen
        self.shift = shift
        self.nfft = nfft
        
    def __call__(self, x):
        window = torch.hann_window(self.winlen, periodic=True, device=x.device)
        X = torch.stft(x, self.nfft, hop_length=self.shift, win_length=self.winlen, window=window, return_complex=True)
        return X
    

class iSTFT():
    def __init__(self, winlen, shift, nfft):
        self.winlen = winlen
        self.shift = shift
        self.nfft = nfft
        
    def __call__(self, X):
        window = torch.hann_window(self.winlen, periodic=True, device=X.device)
        x = torch.istft(X, self.nfft, hop_length=self.shift, win_length=self.winlen, window=window)
        return x