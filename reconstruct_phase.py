import numpy as np
import soundfile as sf
import torch
import argparse
import os

from src.data import postprocess, zero_pad
from src.models import AIGCNN
from src.utils import STFT, iSTFT
from src.algorithms import GLA, DeGLI


def main(file, output_dir, algorithm, n_iter, degli_checkpoint):
    SAMPLING_RATE = 22050
    WINLEN = 1024
    NFFT  = 1024
    SHIFT = 256

    NUM_CH = 32

    stft  = STFT(WINLEN, SHIFT, NFFT)
    istft = iSTFT(WINLEN, SHIFT, NFFT)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data, _ = sf.read(file)
    # If multiple channels
    if len(data.shape) > 1:
        data = data.sum(1)
    # Prepare data
    data /= np.max(np.abs(data))
    len_x = len(data)
    data = np.concatenate((np.zeros(WINLEN-SHIFT,), data, np.zeros(WINLEN-SHIFT,)), axis=0)
    data = zero_pad(data, WINLEN, SHIFT)
    x = torch.FloatTensor(data)

    # Get spectrogram and amplitude
    X = stft(x)
    amp = torch.abs(X)

    if algorithm.lower() == 'gla':
        # GLA
        X = GLA(amp, n_iter, stft, istft)

        x_gla = istft(X)
        x_gla = postprocess(x_gla, len_x, WINLEN, SHIFT)
        sf.write(os.path.join(output_dir, 'gla.wav'), x_gla, SAMPLING_RATE)

    elif algorithm.lower() == 'degli':
        DNN = AIGCNN(NUM_CH, kernel_size=(3,5))
        DNN.load_state_dict(torch.load(degli_checkpoint))
        DNN.eval()
        # DeGLI
        X = DeGLI(amp, DNN, n_iter, stft, istft)

        x_degli = istft(X)
        x_degli = postprocess(x_degli, len_x, WINLEN, SHIFT)
        sf.write(os.path.join(output_dir, 'degli.wav'), x_degli, SAMPLING_RATE)




parser = argparse.ArgumentParser(description="Reconstruct the phase of an audio signal, with GLA or DeGLI")

# Add arguments
parser.add_argument('file', type=str, help='Path to the input file')
parser.add_argument('output_dir', type=str, help='Directory to save the output')
parser.add_argument('algorithm', type=str, help='Algorithm to be used: GLA or DeGLI')
parser.add_argument('--n_iter', type=int, default=10, help='Number of iterations (default: 10)')
parser.add_argument('--degli_checkpoint', type=str, default='checkpoints/degli_ljspeech.pth', help='Path to the DeGLI checkpoint (default: checkpoints/degli_ljspeech.pth)')
    
if __name__=="__main__":
    args = parser.parse_args()
    main(args.file, args.output_dir, args.algorithm, args.n_iter, args.degli_checkpoint)