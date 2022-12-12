import tensorflow
from experiment_eval import eval_ben, eval_adv, eval_adv_return_ben_adv

from PWGAN_preprocessor import PWGAN_WhiteBox_Preprocessor, PWGAN_BlackBox_Preprocessor

import pickle

from training_utils import write_dataset, MyTrainer_Autoconfig
import yaml

import numpy as np
    
from art.estimators.speech_recognition import PyTorchDeepSpeech

from utils import word_error_rate

import sys

print(sys.argv)

with open ('../datasets/gt.pkl', 'rb') as fp:
    labels = pickle.load(fp)

eps = float(sys.argv[1])

pwgan_defense = PWGAN_WhiteBox_Preprocessor("cuda")
wer1, bens, advs = eval_adv_return_ben_adv('../datasets/', eps, pwgan_defense, labels, 7)

bens_audio = [ben[0] for ben in bens]
advs_audio = advs
config = "/tmp/wpgan_tmp/libritts_parallel_wavegan.v1/config.yml"
with open(config) as f:
    config = yaml.load(f, Loader=yaml.Loader)
write_dataset(bens_audio, advs_audio, "/jet/home/rni/ECE_STORAGE/785-proj/tmp_data" ,config, pwgan_defense.pwgan.model.mean.cpu().numpy(), pwgan_defense.pwgan.model.scale.cpu().numpy())
trainer = MyTrainer_Autoconfig("/jet/home/rni/ECE_STORAGE/785-proj/tmp_data", pwgan_defense.pwgan.model, "/tmp/wpgan_tmp/libritts_parallel_wavegan.v1/checkpoint-400000steps.pkl", config, 50, device='cuda')
trainer.run()
wer2, bens, advs = eval_adv_return_ben_adv('../datasets/', eps, pwgan_defense, labels, 7)
print("Trained preprocessor WER: {wer}")

ds2 = PyTorchDeepSpeech(pretrained_model="librispeech", clip_values=[-1, 1])

wers_adv = []
for i in range(len(advs)):
    adv_pred = ds2.predict(x = advs[i], batch_size= 1)
    wers_adv.append(word_error_rate(labels[i], adv_pred[0]))
    
wer3 = np.sum([error for error, _ in wers_adv]) / np.sum([length for _, length in wers_adv])
print(f"Original WER (eps: {eps}): {wer1}")
print(f"Trained preprocessor WER: {wer2}")
print(f"Pretrained WER: {wer3}")