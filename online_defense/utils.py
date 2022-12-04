from torchaudio.functional import melscale_fbanks
import torch


def logmelfilterbank_torch(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-10,
    log_base=10.0,
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
        window=torch.hann_window(win_length),
        pad_mode="reflect",
        return_complex=True
    )
    spc = torch.abs(x_stft).T  # (#frames, #bins)

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

    # return spc, mel_basis
    # return mel_basis
    mel = torch.maximum(torch.FloatTensor([eps]), torch.mm(spc, mel_basis))
    # return mel

    if log_base is None:
        return torch.log(mel)
    elif log_base == 10.0:
        return torch.log10(mel)
    elif log_base == 2.0:
        return torch.log2(mel)
    else:
        raise ValueError(f"{log_base} is not supported.")
