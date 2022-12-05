#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download pretrained Parallel WaveGAN into pretrained_model directory
As part of 11-785 Course Project
Author: Ronghao Ni (ronghaon)
Last revised Nov. 2022
"""

from parallel_wavegan.utils import download_pretrained_model
download_pretrained_model("libritts_parallel_wavegan.v1", "pretrained_model")