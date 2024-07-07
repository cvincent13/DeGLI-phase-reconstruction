import torch


def GLA(amp, n_iter, stft, istft):
    """
        Griffin-Lim algorithm for Phase Reconstruction: returns complex spectrogram X with phase from the given amplitude.

        params:
        - amp: amplitude of the signal to be reconstructed
        - n_iter: number of iteration of the algorithm
        - stft: STFT function
        - istft: inverse of the STFT function
    """
    # Random phase initialization
    random_phase = torch.rand(amp.size())*2*torch.pi - torch.pi
    X_0 = amp * torch.exp(1j*random_phase)

    # GLA iterations
    X = X_0
    for i in range(n_iter):
        Y = amp * torch.sgn(X)
        Z = stft(istft(Y))
        X = Z

    # Final projection
    X = amp * torch.sgn(X)
    return X


def DeGLI(amp, model, n_iter, stft, istft):
    """
        Deep Griﬃn–Lim Iteration algorithm for Phase Reconstruction: returns complex spectrogram X with phase from the given amplitude.

        params:
        - amp: amplitude of the signal to be reconstructed
        - model: DNN model
        - n_iter: number of iteration of the algorithm
        - stft: STFT function
        - istft: inverse of the STFT function
    """
    # Random phase initialization
    random_phase = torch.rand(amp.size())*2*torch.pi - torch.pi
    X_0 = amp * torch.exp(1j*random_phase)

    X = X_0.unsqueeze(0)
    amp_batch = amp.unsqueeze(0)

    # DeGLI iterations
    for i in range(n_iter):
        Y = amp_batch * torch.sgn(X)
        Z = stft(istft(Y))

        X_re = torch.real(X).unsqueeze(1)
        X_im = torch.imag(X).unsqueeze(1)
        Y_re = torch.real(Y).unsqueeze(1)
        Y_im = torch.imag(Y).unsqueeze(1)
        Z_re = torch.real(Z).unsqueeze(1)
        Z_im = torch.imag(Z).unsqueeze(1)
        feat_conc_re  = torch.cat([X_re, Y_re, Z_re], dim=1)
        feat_conc_im  = torch.cat([X_im, Y_im, Z_im], dim=1)

        with torch.no_grad():
            out_dnn = model(feat_conc_re, feat_conc_im, amp_batch)
            X = Z - out_dnn

    # Final projection
    X = amp * torch.sgn(X.squeeze())
    return X