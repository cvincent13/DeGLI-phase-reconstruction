# DeGLI: Audio Phase Reconstruction

Pytorch implementation of the paper [Deep Griﬃn–Lim Iteration: Trainable Iterative Phase Reconstruction Using Neural Network](https://ieeexplore.ieee.org/document/9242279).

Official Tensorflow implementation: https://sites.google.com/view/yoshiki-masuyama/degli


## Reconstruct the phase of an audio signal

`python reconstruct_phase.py [-h] [--n_iter N_ITER] [--degli_checkpoint DEGLI_CHECKPOINT] file output_dir algorithm`

Using the Griffin-Lim Algorithm (GLA): `python reconstruct_phase.py --n_iter 100 [file] [output_dir] gla`

Using DeGLI with trained checkpoint: `python reconstruct_phase.py --n_iter 10 --degli_checkpoint checkpoints/degli_ljspeech.pth [file] [output_dir] degli`


## Training on LJSpeech

1. Download [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) in `make_data`
2. Prepare data for training: `python -m make_data.prepare_LJspeech`
3. Adjust parameters in `train.py`
4. Run `python train.py`
