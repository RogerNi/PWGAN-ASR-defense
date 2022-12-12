import os, re
import numpy as np
from art.attacks.evasion import ProjectedGradientDescent
import soundfile as sf
import torch
from utils import word_error_rate
from art.estimators.speech_recognition import PyTorchDeepSpeech

import sys, os
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr

def eval_ben(benign_folder, preprocessor, labels):
    torch.cuda.empty_cache()
    files = [f for f in os.listdir(benign_folder) if f.endswith(".wav")]
    files = sorted(files, key=lambda f: int(re.sub('\D', '', f)))
    ds2 = PyTorchDeepSpeech(pretrained_model="librispeech", clip_values=[-1, 1], preprocessing_defences=preprocessor)
    
    bens = [os.path.join(benign_folder, f) for f in files]
    bens_audio = [sf.read(f) for f in bens]
    
    wers_ben = []
    for id, audio in tqdm(enumerate(bens_audio)):
        audio = audio[0]
        ben_pred = ds2.predict(x = audio, batch_size= 1)

        # print(recons_res)
#         gt_len = len(strings[id].decode().split())
        # wers.append((wer(strings[id].decode(), recons_res[0]), gt_len))
        wers_ben.append(word_error_rate(labels[id], ben_pred[0]))

    print(wers_ben)
    # return np.sum([wer * weight for wer, weight in wers]) / np.sum([weight for _, weight in wers]), np.average([wer for wer, _ in wers])
    return np.sum([error for error, _ in wers_ben]) / np.sum([length for _, length in wers_ben])

def eval_adv(benign_folder, eps, preprocessor, labels, max_iter=30):
    torch.cuda.empty_cache()
    files = [f for f in os.listdir(benign_folder) if f.endswith(".wav")]
    files = sorted(files, key=lambda f: int(re.sub('\D', '', f)))
    ds2 = PyTorchDeepSpeech(pretrained_model="librispeech", clip_values=[-1, 1], preprocessing_defences=preprocessor)
    pgd = ProjectedGradientDescent(ds2, norm='inf', eps=eps, eps_step=eps/5, max_iter=max_iter, batch_size=1)
    
    bens = [os.path.join(benign_folder, f) for f in files]
    bens_audio = [sf.read(f) for f in bens]
    
    wers_adv = []
    for id, audio in tqdm(enumerate(bens_audio)):
        audio = audio[0]
        # with redirect_stdout(open(os.devnull, "w")), redirect_stderr(open(os.devnull, "w")):
        adv = pgd.generate(np.array([audio]))
            # adv = pgd.generate(audio)
        adv_pred = ds2.predict(x = adv, batch_size= 1)

        # print(recons_res)
#         gt_len = len(strings[id].decode().split())
        # wers.append((wer(strings[id].decode(), recons_res[0]), gt_len))
        wers_adv.append(word_error_rate(labels[id], adv_pred[0]))

    print(wers_adv)
    # return np.sum([wer * weight for wer, weight in wers]) / np.sum([weight for _, weight in wers]), np.average([wer for wer, _ in wers])
    return np.sum([error for error, _ in wers_adv]) / np.sum([length for _, length in wers_adv])

def eval_adv_return_ben_adv(benign_folder, eps, preprocessor, labels, max_iter=30):
    torch.cuda.empty_cache()
    files = [f for f in os.listdir(benign_folder) if f.endswith(".wav")]
    files = sorted(files, key=lambda f: int(re.sub('\D', '', f)))
    ds2 = PyTorchDeepSpeech(pretrained_model="librispeech", clip_values=[-1, 1], preprocessing_defences=preprocessor)
    pgd = ProjectedGradientDescent(ds2, norm='inf', eps=eps, eps_step=eps/5, max_iter=max_iter, batch_size=1)
    
    bens = [os.path.join(benign_folder, f) for f in files]
    bens_audio = [sf.read(f) for f in bens]
    advs_audio = []
    
    wers_adv = []
    for id, audio in tqdm(enumerate(bens_audio)):
        audio = audio[0]
        # with redirect_stdout(open(os.devnull, "w")), redirect_stderr(open(os.devnull, "w")):
        adv = pgd.generate(np.array([audio]))
        advs_audio.append(adv)
            # adv = pgd.generate(audio)
        adv_pred = ds2.predict(x = adv, batch_size= 1)

        # print(recons_res)
#         gt_len = len(strings[id].decode().split())
        # wers.append((wer(strings[id].decode(), recons_res[0]), gt_len))
        wers_adv.append(word_error_rate(labels[id], adv_pred[0]))

    print(wers_adv)
    # return np.sum([wer * weight for wer, weight in wers]) / np.sum([weight for _, weight in wers]), np.average([wer for wer, _ in wers])
    return np.sum([error for error, _ in wers_adv]) / np.sum([length for _, length in wers_adv]), bens_audio, advs_audio