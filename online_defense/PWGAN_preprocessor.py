#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PWGAN_WhiteBox_Preprocessor and PWGAN_BlackBox_Preprocessor, 
compatible with ART preprocessor interface,
As part of 11-785 Course Project
Author: Ronghao Ni (ronghaon)
Last revised Dec. 2022
"""

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from PWGAN_defense import PWGAN_Defense_auto_ckpt
from typing import Optional, Tuple
import torch
import numpy as np

class PWGAN_WhiteBox_Preprocessor(PreprocessorPyTorch):
    def __init__(self, device):
        """Parallel WaveGAN used as a whitebox preprocessor compatible with ART defense.
        Autograd of PyTorch is enabled to calculate backward gradient. All computations are
        performed using Torch operations.

        Args:
            device (str): Name of the device where this preprocessor should be on
        """
        super().__init__(
            device_type=device,
            is_fitted=True,
            apply_fit=False,
            apply_predict=True,
        )
        self.pwgan = PWGAN_Defense_auto_ckpt().to(device)
        
    def forward(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """Forward an input through the preprocessor
        """
        return self.pwgan(x), y
    
    
class PWGAN_BlackBox_Preprocessor(PWGAN_WhiteBox_Preprocessor):
    def __init__(self, device):
        super().__init__(device)
        
    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Estimate BlackBox Gradient using BPDA with the identity function.

        Args:
            x (np.ndarray): Forward input
            grad (np.ndarray): Backward gradient

        Returns:
            np.ndarray: Estimated gradient
        """
        return grad