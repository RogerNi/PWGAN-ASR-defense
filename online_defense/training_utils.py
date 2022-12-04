from parallel_wavegan.bin.train import Trainer, Collater
from tqdm import tqdm
import logging
import torch
from torch.utils.data import DataLoader
import parallel_wavegan
from parallel_wavegan.losses import DiscriminatorAdversarialLoss
from parallel_wavegan.losses import FeatureMatchLoss
from parallel_wavegan.losses import GeneratorAdversarialLoss
from parallel_wavegan.losses import MelSpectrogramLoss
from parallel_wavegan.losses import MultiResolutionSTFTLoss
from parallel_wavegan.datasets import AudioMelDataset
from parallel_wavegan.utils import read_hdf5

class MyTrainer(Trainer):
    def __init__(self, steps, epochs, data_loader, sampler, model, criterion, optimizer, scheduler, config, nepochs, device=...):
        super().__init__(steps, epochs, data_loader, sampler, model, criterion, optimizer, scheduler, config, device)
        self.nepochs = nepochs
    
    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()
            self.nepochs -= 1

            # check whether training is finished
            if self.nepochs <= 0:
                break

        self.tqdm.close()
        logging.info("Finished training.")
        
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
        
    def get_generator(self):
        return self.model['generator']
        
class MyTrainer_Autoconfig(MyTrainer):
    def __init__(self, data_root, generator, discriminator, config, nepochs, device=...):
        
        train_dataset = AudioMelDataset(
            root_dir=data_root,
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
            aux_context_window=config["generator_params"].get("aux_context_window", 0),
            # keep compatibility
            use_noise_input=config.get("generator_type", "ParallelWaveGANGenerator")
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
        if config.get("use_subband_stft_loss", False):  # keep compatibility
            assert config["generator_params"]["out_channels"] > 1
            criterion["sub_stft"] = MultiResolutionSTFTLoss(
                **config["subband_stft_loss_params"],
            ).to(device)
        else:
            config["use_subband_stft_loss"] = False
        if config.get("use_feat_match_loss", False):  # keep compatibility
            criterion["feat_match"] = FeatureMatchLoss(
                # keep compatibility
                **config.get("feat_match_loss_params", {}),
            ).to(device)
        else:
            config["use_feat_match_loss"] = False
        if config.get("use_mel_loss", False):  # keep compatibility
            if config.get("mel_loss_params", None) is None:
                criterion["mel"] = MelSpectrogramLoss(
                    fs=config["sampling_rate"],
                    fft_size=config["fft_size"],
                    hop_size=config["hop_size"],
                    win_length=config["win_length"],
                    window=config["window"],
                    num_mels=config["num_mels"],
                    fmin=config["fmin"],
                    fmax=config["fmax"],
                ).to(device)
            else:
                criterion["mel"] = MelSpectrogramLoss(
                    **config["mel_loss_params"],
                ).to(device)
        else:
            config["use_mel_loss"] = False
            
        model = {
            "generator": generator.to(device),
            "discriminator": discriminator.to(device),
        }
        
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
        
        generator_scheduler_class = getattr(
            torch.optim.lr_scheduler,
            # keep compatibility
            config.get("generator_scheduler_type", "StepLR"),
        )
        discriminator_scheduler_class = getattr(
            torch.optim.lr_scheduler,
            # keep compatibility
            config.get("discriminator_scheduler_type", "StepLR"),
        )
        scheduler = {
            "generator": generator_scheduler_class(
                optimizer=optimizer["generator"],
                **config["generator_scheduler_params"],
            ),
            "discriminator": discriminator_scheduler_class(
                optimizer=optimizer["discriminator"],
                **config["discriminator_scheduler_params"],
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
            scheduler=scheduler, 
            config=config, 
            nepochs=nepochs, 
            device=device)