import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


from src.data import AudioDataset
from src.models import AIGCNN
from src.utils import STFT, iSTFT


# Config
SAMPLING_RATE = 22050
WINLEN = 1024
NFFT  = 1024
SHIFT = 256

NUM_FREQ = int(np.floor(NFFT/2))+1
NUM_FRAMES = 101

NUM_CH = 32
EPOCHS = 300
BATCH_SIZE = 2
LR = 4e-4
SNR_MAX = 12.
SNR_MIN = -6.

SAVE_CHECKPOINT_NAME = "checkpoints/new_checkpoint.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

stft  = STFT(WINLEN, SHIFT, NFFT)
istft = iSTFT(WINLEN, SHIFT, NFFT)


# Model
DNN = AIGCNN(NUM_CH, kernel_size=(3,5)).to(device)

# Data
data_folder = 'data/'
train_files = [os.path.join(data_folder, 'train', name) for name in os.listdir(os.path.join(data_folder, 'train'))]
valid_files = [os.path.join(data_folder, 'valid', name) for name in os.listdir(os.path.join(data_folder, 'valid'))]
test_files = [os.path.join(data_folder, 'test', name) for name in os.listdir(os.path.join(data_folder, 'test'))]
    

train_dataset = AudioDataset(train_files, stft)
valid_dataset = AudioDataset(valid_files, stft)
test_dataset = AudioDataset(test_files, stft)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training
def phase_sensitive_mse(x_star, x_est):
    return (torch.abs(x_star-x_est)**2).mean()

optimizer = torch.optim.Adam(DNN.parameters(), LR)
lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

train_losses = []
valid_losses = []

for ep in range(EPOCHS):
    ep_loss = 0
    ep_val_loss = 0
    for (X, amps) in tqdm(train_loader, desc='Training...'):
        DNN.train()
        optimizer.zero_grad()
        # Add gaussian noise
        adn = np.random.randn(X.size(0), NUM_FREQ, NUM_FRAMES) + 1j*np.random.randn(X.size(0), NUM_FREQ, NUM_FRAMES)
        norm_adn = np.linalg.norm(adn, ord='fro', axis = (1,2))
        adn = adn / norm_adn[:, np.newaxis, np.newaxis] * 10.**(-1.*(np.random.rand()*(SNR_MAX-SNR_MIN)+SNR_MIN)/20)
        noise = torch.tensor(adn, dtype=torch.complex64)
        noise = noise*torch.linalg.norm(X, ord='fro', dim=[-2,-1], keepdim=True)

        X = X.to(device)
        amps = amps.to(device)
        noise = noise.to(device)
        X_noise = X + noise
        
        # Projections
        Y = amps * torch.sgn(X_noise)
        Z = stft(istft(Y))

        # DNN
        X_re = torch.real(X_noise).unsqueeze(1)
        X_im = torch.imag(X_noise).unsqueeze(1)
        Y_re = torch.real(Y).unsqueeze(1)
        Y_im = torch.imag(Y).unsqueeze(1)
        Z_re = torch.real(Z).unsqueeze(1)
        Z_im = torch.imag(Z).unsqueeze(1)
        feat_conc_re  = torch.cat([X_re, Y_re, Z_re], dim=1)
        feat_conc_im  = torch.cat([X_im, Y_im, Z_im], dim=1)
        out_dnn = DNN(feat_conc_re, feat_conc_im, amps)
        X_out = Z - out_dnn

        loss = phase_sensitive_mse(X, X_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(DNN.parameters(), max_norm=100.)
        optimizer.step()

        ep_loss += loss.item()

    

    for (X, amps) in tqdm(valid_loader, desc='Validating...'):
        DNN.eval()
        # Add noise
        adn = np.random.randn(X.size(0), NUM_FREQ, NUM_FRAMES) + 1j*np.random.randn(X.size(0), NUM_FREQ, NUM_FRAMES)
        norm_adn = np.linalg.norm(adn, ord='fro', axis = (1,2))
        adn = adn / norm_adn[:, np.newaxis, np.newaxis] * 10.**(-1.*(np.random.rand()*(SNR_MAX-SNR_MIN)+SNR_MIN)/20)
        noise = torch.tensor(adn, dtype=torch.complex64)
        noise = noise*torch.linalg.norm(X, ord='fro', dim=[-2,-1], keepdim=True)

        X = X.to(device)
        amps = amps.to(device)
        noise = noise.to(device)
        X_noise = X + noise

        # Projections
        Y = amps * torch.sgn(X_noise)
        Z = stft(istft(Y))

        # DNN
        X_re = torch.real(X_noise).unsqueeze(1)
        X_im = torch.imag(X_noise).unsqueeze(1)
        Y_re = torch.real(Y).unsqueeze(1)
        Y_im = torch.imag(Y).unsqueeze(1)
        Z_re = torch.real(Z).unsqueeze(1)
        Z_im = torch.imag(Z).unsqueeze(1)
        feat_conc_re  = torch.cat([X_re, Y_re, Z_re], dim=1)
        feat_conc_im  = torch.cat([X_im, Y_im, Z_im], dim=1)
        with torch.no_grad():
            out_dnn = DNN(feat_conc_re, feat_conc_im, amps)
            X_out = Z - out_dnn
            valid_loss = phase_sensitive_mse(X, X_out)
            
        ep_val_loss += valid_loss.item()
        
    lr_scheduler.step()
    ep_loss = ep_loss/len(train_loader)
    ep_val_loss = ep_val_loss/len(valid_loader)
    train_losses.append(ep_loss)
    valid_losses.append(ep_val_loss)
    print(f'Epoch {ep+1}: Train loss = {ep_loss:.6f}, Validation loss = {ep_val_loss:.6f}')

torch.save(DNN.state_dict(), SAVE_CHECKPOINT_NAME)