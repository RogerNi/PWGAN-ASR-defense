#!/bin/bash
# Script to process all audio samples through Parallel WaveGAN
# As part of 11-785 Course Project
# Ronghao Ni (ronghaon@andrew.cmu.edu)
# Last revised Nov. 2022

Help()
{
   # Display Help
   echo "Process all audio samples through Parallel WaveGAN."
   echo
   echo "Syntax: all_process.bash [-h] <PWGAN_path> <samples_path>"
   echo "Arguments:"
   echo "   <PWGAN_path>    Path to the root of Parallel WaveGAN Repo."
   echo "   <samples_path>  Path to the root folder of all samples to process."
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

./sox_batch_sampling_rate.bash $2 $2/outputdir 24000
./gan_reconstruct.bash $1 $2/outputdir $2/reconstructed
./sox_batch_sampling_rate.bash $2/reconstructed $2/reconstructed_low_sampled 16000