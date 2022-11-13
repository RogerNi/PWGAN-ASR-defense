#!/bin/bash
# Script to reconstruct audio samples using Parallel WaveGAN
# As part of 11-785 Course Project
# Ronghao Ni (ronghaon@andrew.cmu.edu)
# Last revised Nov. 2022

Help()
{
   # Display Help
   echo "Reconstruct audio samples using Parallel WaveGAN."
   echo
   echo "Syntax: gan_reconstruct.bash [-h] <PWGAN_path> <input_path> <output_path>"
   echo "Arguments:"
   echo "   <PWGAN_path>    Path to the root of Parallel WaveGAN Repo."
   echo "   <input_path>    Path to input folder."
   echo "   <output_path>   Path to output folder (creat if missing)."
   echo ""
   echo "Option:"
   echo "   h               Print this Help message."
   echo ""
}

################################################################################
# Get the options
while getopts ":h" option; do
   case $option in
      h)
         Help
         exit;;
   esac
done


rm -rf $1/dump/sample
parallel-wavegan-preprocess  --config $1/pretrained_model/libritts_parallel_wavegan.v1.long/config.yml --rootdir $2 --dumpdir $1/dump/sample/raw
parallel-wavegan-normalize --config $1/pretrained_model/libritts_parallel_wavegan.v1.long/config.yml --rootdir $1/dump/sample/raw --dumpdir $1/dump/sample/norm --stats $1/pretrained_model/libritts_parallel_wavegan.v1.long/stats.h5 
mkdir -p $3
parallel-wavegan-decode --checkpoint $1/pretrained_model/libritts_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl  --dumpdir $1/dump/sample/norm --outdir $3