from torchaudio.functional import melscale_fbanks
import torch


def logmelfilterbank_torch(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window=None,
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-10,
    log_base=10.0,
    device='cpu'
):
    """Compute log-Mel filterbank feature using Torch operations.
    Based on Parallel WaveGAN Implementation but replace all operation 
    to Torch operations to support faster computation and autograd.
    """
    # get amplitude spectrogram
    x_stft = torch.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
        return_complex=True
    )
    spc = torch.abs(x_stft).T.float()  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = melscale_fbanks(
        sample_rate=sampling_rate,
        n_freqs=fft_size//2 + 1,
        n_mels=num_mels,
        f_min=fmin,
        f_max=fmax,
        norm='slaney',
        mel_scale='slaney'
    )

    mel_basis = mel_basis.to(device)

    # return spc, mel_basis
    # return mel_basis
    mel = torch.maximum(torch.FloatTensor([eps]).to(
        device), torch.mm(spc.squeeze(), mel_basis))
    # return mel

    if log_base is None:
        return torch.log(mel)
    elif log_base == 10.0:
        return torch.log10(mel)
    elif log_base == 2.0:
        return torch.log2(mel)
    else:
        raise ValueError(f"{log_base} is not supported.")


def word_error_rate(y, y_pred):
    import numpy as np
    """
    Return the word error rate for a batch of transcriptions. 
    Orignial code: https://martin-thoma.com/word-error-rate-calculation/
    """
    if isinstance(y, bytes):
        y = y.decode("utf-8")
    elif not isinstance(y, str):
        raise TypeError(f"y is of type {type(y)}, expected string or bytes")
    if isinstance(y_pred, bytes):
        y_pred = y_pred.decode("utf-8")
    elif not isinstance(y_pred, str):
        raise TypeError(
            f"y_pred is of type {type(y_pred)}, expected string or bytes")
    reference = y.split()
    hypothesis = y_pred.split()

    r_length = len(reference)
    h_length = len(hypothesis)
    matrix = np.zeros((r_length + 1, h_length + 1))
    for i in range(r_length + 1):
        for j in range(h_length + 1):
            if i == 0:
                matrix[0][j] = j
            elif j == 0:
                matrix[i][0] = i
    for i in range(1, r_length + 1):
        for j in range(1, h_length + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                substitute = matrix[i - 1][j - 1] + 1
                insertion = matrix[i][j - 1] + 1
                deletion = matrix[i - 1][j] + 1
                matrix[i][j] = min(substitute, insertion, deletion)
    return (matrix[r_length][h_length], r_length)


# alias for word_error_rate
WER = word_error_rate
