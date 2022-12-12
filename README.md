# PWGAN-ASR-defense

## Introduction

This is the codebase for the course project of 11-785 Introduction to Deep Learning. This includes all the codes needed to replicate our experiments. 

## Contributor

Ronghao Ni (RogerNi) (ronghaon@andrew.cmu.edu): all codes (that were used to run experiments mentioned in our reports)

## File structures

`offline_defense`: all codes for running our offline defense experiments

`online_defense`: all codes for running our online defense experiments

`dataset.tar.gz`: this file should be extracted into `./datasets` folder in the root directory in order for the experiments to find the required evaluation data. This `gz` file includes 100 utterances from LibriSpeech dataset and also the corresponding transcripts.

## Setup

Please follow `./online_defense/setup.ipynb` to setup the environment in order to run the experiment. Please also notice that some codes in Adversarial Robustness Toolbox need to be modified to make it compatible with the latest `Deep Speech 2` and `Pytorch Ligtning`


In `~/.local/lib/python3.8/site-packages/art/estimators/speech_recognition/pytorch_deep_speech.py`, change line 149 to: 
```python
elif str(DeepSpeech.__base__) == "<class 'pytorch_lightning.core.module.LightningModule'>":
```

, change line 384 to: 
```python
outputs, output_sizes, _ = self._model(
```

, and change line 485 to: 
```python   
outputs, output_sizes, _ = self._model(inputs.to(self._device), input_sizes.to(self._device))
```

Besides, change `~/.local/lib/python3.9/site-packages/art/estimators/pytorch.py`, line 249 to:
```python
gradients = gradients[0]
gradients = torch.tensor(gradients, device=self._device)
gradients = gradients[None,:]
```

## Example to run experiments

### Run online defense (without BPDA)

```python
import tensorflow
from experiment_eval import eval_ben, eval_adv

from PWGAN_preprocessor import PWGAN_WhiteBox_Preprocessor, PWGAN_BlackBox_Preprocessor
import pickle
pwgan_defense = PWGAN_WhiteBox_Preprocessor("cuda")

with open ('../datasets/gt.pkl', 'rb') as fp:
    labels = pickle.load(fp)

bb_results = []
for eps in [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2]:
    bb_results.append(eval_adv('../datasets/', eps, pwgan_defense, labels))
    print(bb_results)

```

### Run online defense (with BPDA)

```python
import tensorflow
from experiment_eval import eval_ben, eval_adv

from PWGAN_preprocessor import PWGAN_WhiteBox_Preprocessor, PWGAN_BlackBox_Preprocessor
import pickle
pwgan_defense = PWGAN_BlackBox_Preprocessor("cuda")

with open ('../datasets/gt.pkl', 'rb') as fp:
    labels = pickle.load(fp)

bb_results = []
for eps in [0.0001, 0.001, 0.005, 0.01, 0.1, 0.2]:
    bb_results.append(eval_adv('../datasets/', eps, pwgan_defense, labels))
    print(bb_results)
```

### Run active learning (see `./online_defense/active_learning.py`)

```python
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
```

## Experiment results

### Offline defense

The results of offline defense are not included here as these are done before mid-term report and we used a different codebase to do the offline defense. The experiment is done in a `ipynb` and can be provided upon requests.

### Online defense

Results can be found in `./online_defense/run_exp.ipynb` and `./online_defense/run_exp2.ipynb`

### Active learning

Results can be found in `./online_defense/active_learning.log`