from torchaudio.functional import Resample
from functools import partial
import torch
from utils import logmelfilterbank_torch
from parallel_wavegan.utils import load_model
import os, tempfile
from parallel_wavegan.utils import download_pretrained_model


class PWGAN_Defense(torch.nn.Module):
    def __init__(self, config, ckpt_path):
        super().__init__()
        self.pre_resample = Resample(16000, 24000)
        self.post_resample = Resample(24000, 16000)
        self.log_mel_feature = partial(
            logmelfilterbank_torch,             
            sampling_rate=config["sampling_rate"],
            hop_size=config["hop_size"],
            fft_size=config["fft_size"],
            win_length=config["win_length"],
            window=config["window"],
            num_mels=config["num_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
            log_base=config.get("log_base", 10.0)
        )
        model = load_model(ckpt_path, config)
        model.remove_weight_norm()
        model = model.eval()
        self.pwgan = model
        
    def forward(self, x):
        x = self.pre_resample(x)
        x = self.log_mel_feature(x)
        x = self.model.inference(x, normalize_before=True).view(-1)
        return self.post_resample(x)
        
        
class PWGAN_Defense_auto_config(PWGAN_Defense):
    def __init__(self, ckpt_path):
        dirname = os.path.dirname(ckpt_path)
        config = os.path.join(dirname, "config.yml")
        super().__init__(config, ckpt_path)
        
class PWGAN_Defense_auto_ckpt(PWGAN_Defense_auto_config):
    def __init__(self):
        tmp_dir = os.path.join(tempfile.gettempdir(), "wpgan_tmp")
        try:
            os.makedirs(tmp_dir)
            download_pretrained_model("1zHQl8kUYEuZ_i1qEFU6g2MEu99k3sHmR", tmp_dir)
        except:
            pass
        super().__init__(os.path.join(tmp_dir, "checkpoint-400000steps.pkl"))