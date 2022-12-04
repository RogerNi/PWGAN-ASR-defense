#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PWGAN_Defense class, compatible with PyTorch nn.Module 
(also in two varient versions supporting automatic config-loading and checkpoint-loading)
As part of 11-785 Course Project
Author: Ronghao Ni (ronghaon)
Last revised Dec. 2022
"""


from torchaudio.functional import resample
from functools import partial
import torch
from utils import logmelfilterbank_torch
from parallel_wavegan.utils import load_model
import os, tempfile
from parallel_wavegan.utils import download_pretrained_model
import yaml


class PWGAN_Defense(torch.nn.Module):
    def __init__(self, config, ckpt_path):
        super().__init__()
        self.pre_resample = partial(resample, orig_freq=16000, new_freq=24000)
        self.post_resample = partial(resample, orig_freq=24000, new_freq=16000)
        self.window_tensor = torch.hann_window(config["win_length"])
        self.config = config
        self.log_mel_feature = partial(
            logmelfilterbank_torch,             
            sampling_rate=config["sampling_rate"],
            hop_size=config["hop_size"],
            fft_size=config["fft_size"],
            win_length=config["win_length"],
            window=self.window_tensor,
            num_mels=config["num_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
            log_base=config.get("log_base", 10.0)
        )
        model = load_model(ckpt_path, config)
        model.remove_weight_norm()
        model = model.eval()
        self.model = model
        
    def forward(self, x):
        x = self.pre_resample(x)
        x = self.log_mel_feature(x)
        x = self.model.inference(x, normalize_before=True).view(-1)
        return self.post_resample(x).unsqueeze(0)
    
    def to(self, device):
        new_self = super(PWGAN_Defense, self).to(device)
        self.window_tensor = self.window_tensor.to(device)
        config = self.config
        self.log_mel_feature = partial(
            logmelfilterbank_torch,             
            sampling_rate=config["sampling_rate"],
            hop_size=config["hop_size"],
            fft_size=config["fft_size"],
            win_length=config["win_length"],
            window=self.window_tensor,
            num_mels=config["num_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
            log_base=config.get("log_base", 10.0),
            device=device
        )

        return new_self
        
        
class PWGAN_Defense_auto_config(PWGAN_Defense):
    def __init__(self, ckpt_path):
        dirname = os.path.dirname(ckpt_path)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        super().__init__(config, ckpt_path)
        
class PWGAN_Defense_auto_ckpt(PWGAN_Defense_auto_config):
    def __init__(self):
        tmp_dir = os.path.join(tempfile.gettempdir(), "wpgan_tmp")
        try:
            os.makedirs(tmp_dir)
            download_pretrained_model("libritts_parallel_wavegan.v1", tmp_dir)
        except Exception as e:
            pass
        super().__init__(os.path.join(tmp_dir, "libritts_parallel_wavegan.v1", "checkpoint-400000steps.pkl"))