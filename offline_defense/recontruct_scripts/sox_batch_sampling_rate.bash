#!/bin/bash
# Script to batch convert sampling rates using sox
# As part of 11-785 Course Project
# Ronghao Ni (ronghaon@andrew.cmu.edu)
# Last revised Nov. 2022

Help()
{
   # Display Help
   echo "Batch convert sampling rates using sox."
   echo
   echo "Syntax: sox_batch_sampling_rate.bash [-h] <input_path> <output_path> <sampling_rate>"
   echo "Arguments:"
   echo "   <input_path>    Path to input folder."
   echo "   <output_path>   Path to output folder (created if missing)."
   echo "   <sampling_rate> Sampling rates to convert to"
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


mkdir -p $2
for i in $1/*.wav; do
    o=$2/${i#$1/}
    sox "$i" -r $3 "${o%.wav}.wav"
done