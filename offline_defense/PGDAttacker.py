from art.attacks.evasion import ProjectedGradientDescent
import os
import re
import numpy as np
from scipy.io import wavfile
from utils.ART_util import word_error_rate
import subprocess
import pathlib


class PGDAttacker:
    def __init__(self, model, norm, eps, eps_step, max_iter, dataset_path, labels, tmp_path, pwgan_dir):
        self.model = model
        self.pgd = ProjectedGradientDescent(
            self.model, norm=norm, eps=eps, eps_step=eps_step, max_iter=max_iter)
        self.dataset = dataset_path
        self.labels = labels
        self.tmp_path = tmp_path
        self.pwgan_dir = pwgan_dir # this is the root of pretrained_models

    def no_defense_wer(self):
        output_dir = os.path.join(self.tmp_path, "adversarial")

        files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]

        files = sorted(files, key=lambda f: int(re.sub('\D', '', f)))

        adv = [os.path.join(output_dir, f)
               for f in files if f.endswith("adversarial.wav")]

        wers = []
        for id, file in enumerate(adv):
            samplerate, data = wavfile.read(file)
            recons_res = self.model.predict(x=np.array([data]), batch_size=1)
            # print(recons_res)
            gt_len = len(self.labels[id].decode().split())
            # wers.append((wer(strings[id].decode(), recons_res[0]), gt_len))
            wers.append(word_error_rate(self.labels[id], recons_res[0]))

        # print(wers)
        # return np.sum([wer * weight for wer, weight in wers]) / np.sum([weight for _, weight in wers]), np.average([wer for wer, _ in wers])
        return np.sum([error for error, _ in wers]) / np.sum([length for _, length in wers])

    def defense_wer(self):
        output_dir = os.path.join(self.tmp_path, "adversarial", "reconstructed_low_sampled")
        import os
        import re
        import numpy as np
        from scipy.io import wavfile
        files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]

        files = sorted(files, key=lambda f: int(re.sub('\D', '', f)))

        adv = [os.path.join(output_dir, f)
               for f in files if f.endswith("adversarial_gen.wav")]

        wers = []
        for id, file in enumerate(adv):
            samplerate, data = wavfile.read(file)
            data = data.astype(np.float32)
            max_int16 = 2**40
            data = data / max_int16
            recons_res = self.model.predict(x=np.array([data]), batch_size=1)
            # print(recons_res)
            gt_len = len(self.labels[id].decode().split())
            # wers.append((wer(strings[id].decode(), recons_res[0]), gt_len))
            wers.append(word_error_rate(self.labels[id], recons_res[0]))

        # print(wers)
        # return np.sum([wer * weight for wer, weight in wers]) / np.sum([weight for _, weight in wers]), np.average([wer for wer, _ in wers])
        return np.sum([error for error, _ in wers]) / np.sum([length for _, length in wers])

    def attack(self):
        output_dir = os.path.join(self.tmp_path, "adversarial")
        try:
            os.makedirs(output_dir)
        except:
            print("Adversarial examples exist. No attack is performed.")
            return

        files = [f for f in os.listdir(self.dataset) if f.endswith(".wav")]

        files = sorted(files, key=lambda f: int(re.sub('\D', '', f)))

        adv = [os.path.join(self.dataset, f)
               for f in files if f.endswith("benign.wav")]

        index = 0
        for id, file in enumerate(adv):
            samplerate, data = wavfile.read(file)
            adv_sample = (self.pgd.generate(np.array([data])))
            wavfile.write(os.path.join(output_dir, str(index) +
                          "_adversarial.wav"), samplerate, adv_sample.T)
            index += 1

        return index

    def defense(self):
        script_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "reconstruct_scripts")
        CD = 'cd'
        AND = "&&"
        output_dir = os.path.join(self.tmp_path, "adversarial")
        bashCommand = [CD, script_dir, AND, "./all_process.bash", self.pwgan_dir, output_dir]
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        return process.communicate()
