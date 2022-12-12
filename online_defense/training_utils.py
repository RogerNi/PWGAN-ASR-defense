from parallel_wavegan.bin.train import Trainer, Collater
from tqdm import tqdm
import os
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
import parallel_wavegan
from parallel_wavegan.losses import GeneratorAdversarialLoss
from parallel_wavegan.losses import DiscriminatorAdversarialLoss
from parallel_wavegan.losses import MultiResolutionSTFTLoss
from parallel_wavegan.datasets import AudioMelDataset
from parallel_wavegan.utils import read_hdf5, write_hdf5
import librosa
from parallel_wavegan.bin.preprocess import logmelfilterbank

# Disable torch UserWarning caused by Parallel WaveGAN codes
import warnings
warnings.filterwarnings("ignore")


class MyTrainer(Trainer):
    """This trainer is modified from the original implementation in 
    https://github.com/kan-bayashi/ParallelWaveGAN/blob/d5d479209579c20aaa03bace4abcb0b31bc417f5/parallel_wavegan/bin/train.py
    with changes for our tasks.
    """
    def __init__(self, steps, epochs, data_loader, sampler, model, criterion, optimizer, scheduler, config, nepochs, device=...):
        super().__init__(steps, epochs, data_loader, sampler, model,
                         criterion, optimizer, scheduler, config, device)
        self.nepochs = nepochs

    def run(self):
        """Run training."""
        try:
            self.model['generator'].apply_weight_norm()
        except:
            pass
        self.model['generator'].train()
        # self.tqdm = tqdm(
        #     initial=self.steps, desc="[train]"
        # )
        self.pbar = tqdm()
        while True:
            # train one epoch
            self._train_epoch()
            self.nepochs -= 1
            self.pbar.update(1)

            # check whether training is finished
            if self.nepochs <= 0:
                break
        self.pbar.close()

        # self.tqdm.close()
        self.model['generator'].remove_weight_norm()
        self.model['generator'].eval()
        logging.info("Finished training.")
        


    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        #######################
        #      Generator      #
        #######################
        if self.steps > self.config.get("generator_train_start_steps", 0):
            y_ = self.model["generator"](*x)

            # reconstruct the signal from multi-band signal
            if self.config["generator_params"]["out_channels"] > 1:
                y_mb_ = y_
                y_ = self.criterion["pqmf"].synthesis(y_mb_)

            # initialize
            gen_loss = 0.0

            # multi-resolution sfft loss
            if self.config["use_stft_loss"]:
                sc_loss, mag_loss = self.criterion["stft"](y_, y)
                gen_loss += sc_loss + mag_loss
                self.total_train_loss[
                    "train/spectral_convergence_loss"
                ] += sc_loss.item()
                self.total_train_loss[
                    "train/log_stft_magnitude_loss"
                ] += mag_loss.item()

            # subband multi-resolution stft loss
            if self.config["use_subband_stft_loss"]:
                gen_loss *= 0.5  # for balancing with subband stft loss
                y_mb = self.criterion["pqmf"].analysis(y)
                sub_sc_loss, sub_mag_loss = self.criterion["sub_stft"](y_mb_, y_mb)
                gen_loss += 0.5 * (sub_sc_loss + sub_mag_loss)
                self.total_train_loss[
                    "train/sub_spectral_convergence_loss"
                ] += sub_sc_loss.item()
                self.total_train_loss[
                    "train/sub_log_stft_magnitude_loss"
                ] += sub_mag_loss.item()

            # mel spectrogram loss
            if self.config["use_mel_loss"]:
                mel_loss = self.criterion["mel"](y_, y)
                gen_loss += mel_loss
                self.total_train_loss["train/mel_loss"] += mel_loss.item()

            # weighting aux loss
            gen_loss *= self.config.get("lambda_aux", 1.0)

            # adversarial loss
            if self.steps > self.config["discriminator_train_start_steps"]:
                p_ = self.model["discriminator"](y_)
                adv_loss = self.criterion["gen_adv"](p_)
                self.total_train_loss["train/adversarial_loss"] += adv_loss.item()

                # feature matching loss
                if self.config["use_feat_match_loss"]:
                    # no need to track gradients
                    with torch.no_grad():
                        p = self.model["discriminator"](y)
                    fm_loss = self.criterion["feat_match"](p_, p)
                    self.total_train_loss[
                        "train/feature_matching_loss"
                    ] += fm_loss.item()
                    adv_loss += self.config["lambda_feat_match"] * fm_loss

                # add adversarial loss to generator loss
                gen_loss += self.config["lambda_adv"] * adv_loss

            self.total_train_loss["train/generator_loss"] += gen_loss.item()

            # update generator
            self.optimizer["generator"].zero_grad()
            gen_loss.backward()
            if self.config["generator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["generator"].parameters(),
                    self.config["generator_grad_norm"],
                )
            self.optimizer["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_ = self.model["generator"](*x)
            if self.config["generator_params"]["out_channels"] > 1:
                y_ = self.criterion["pqmf"].synthesis(y_)

            # discriminator loss
            p = self.model["discriminator"](y)
            p_ = self.model["discriminator"](y_.detach())
            real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
            dis_loss = real_loss + fake_loss
            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"],
                )
            self.optimizer["discriminator"].step()

        # update counts
        self.steps += 1
        
    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )
        self.pbar.set_description(f"[Train]")

    def get_generator(self):
        return self.model['generator']


class MyTrainer_Autoconfig(MyTrainer):
    def __init__(self, data_path, generator, ckpt_path, config, nepochs, device=...):
        
        # config["generator_optimizer_params"]["lr"] = 1e-6

        train_dataset = AudioMelDataset(
            root_dir=data_path,
            audio_query="*.h5",
            mel_query="*.h5",
            audio_load_fn=lambda x: read_hdf5(x, "wave"),
            mel_load_fn=lambda x: read_hdf5(x, "feats"),
            mel_length_threshold=None,
            allow_cache=config.get("allow_cache", False),  # keep compatibility
        )

        collater = Collater(
            batch_max_steps=config["batch_max_steps"],
            hop_size=config["hop_size"],
            # keep compatibility
            aux_context_window=config["generator_params"].get(
                "aux_context_window", 0),
            # keep compatibility
            use_noise_input=config.get(
                "generator_type", "ParallelWaveGANGenerator")
            in ["ParallelWaveGANGenerator"],
        )

        data_loader = {
            "train": DataLoader(
                dataset=train_dataset,
                shuffle=True,
                collate_fn=collater,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                sampler=None,
                pin_memory=config["pin_memory"],
            ),
        }

        criterion = {
            "gen_adv": GeneratorAdversarialLoss(
                # keep compatibility
                **config.get("generator_adv_loss_params", {})
            ).to(device),
            "dis_adv": DiscriminatorAdversarialLoss(
                # keep compatibility
                **config.get("discriminator_adv_loss_params", {})
            ).to(device),
        }

        if config.get("use_stft_loss", True):  # keep compatibility
            config["use_stft_loss"] = True
            criterion["stft"] = MultiResolutionSTFTLoss(
                **config["stft_loss_params"],
            ).to(device)
        config["use_subband_stft_loss"] = False
        config["use_feat_match_loss"] = False
        config["use_mel_loss"] = False
        
        discriminator_class = getattr(
            parallel_wavegan.models,
            # keep compatibility
            config.get("discriminator_type", "ParallelWaveGANDiscriminator"),
        )

        model = {
            "generator": generator.to(device),
            "discriminator": discriminator_class(
                **config["discriminator_params"],
            ).to(device),
        }
        
        state_dict = torch.load(ckpt_path, map_location="cpu")
        
        model["discriminator"].load_state_dict(
            state_dict["model"]["discriminator"]
        )

        generator_optimizer_class = getattr(
            parallel_wavegan.optimizers,
            # keep compatibility
            config.get("generator_optimizer_type", "RAdam"),
        )
        
        discriminator_optimizer_class = getattr(
            parallel_wavegan.optimizers,
            # keep compatibility
            config.get("discriminator_optimizer_type", "RAdam"),
        )
        
        optimizer = {
            "generator": generator_optimizer_class(
                model["generator"].parameters(),
                **config["generator_optimizer_params"],
            ),
            "discriminator": discriminator_optimizer_class(
                model["discriminator"].parameters(),
                **config["discriminator_optimizer_params"],
            ),
        }

        super().__init__(
            steps=0,
            epochs=0,
            data_loader=data_loader,
            sampler={"train": None, "dev": None},
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            config=config,
            nepochs=nepochs,
            device=device)

def write_dataset(bens, advs, data_path, config, mean, scale):
    assert (len(bens) == len(advs))
    for i in range(len(bens)):
        filepath = os.path.join(data_path, str(i) + ".h5")
        resampled_ben = librosa.resample(bens[i], orig_sr=16000, target_sr=24000)
        resampled_adv = librosa.resample(advs[i], orig_sr=16000, target_sr=24000)
        adv_feature = logmelfilterbank(
            resampled_adv[0],
            sampling_rate=config["sampling_rate"],
            hop_size=config["hop_size"],
            fft_size=config["fft_size"],
            win_length=config["win_length"],
            window=config["window"],
            num_mels=config["num_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
            # keep compatibility
            log_base=config.get("log_base", 10.0),
        )
        adv_feature = (adv_feature - mean) / scale
        write_hdf5(filepath, "wave", resampled_ben)
        write_hdf5(filepath, "feats", adv_feature)
    