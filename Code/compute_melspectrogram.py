"""
Compute the mel-spectrogram for batch of audio files
"""

import argparse
import random
import os
import librosa
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/Users/camillenoufi/Documents/datasets/VocEx-local/local_balanced/', help="Directory with the master valid DAMP-VocEx dataset")


if __name__ == '__main__':

    # Process command line
   args = parser.parse_args()
   assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
   master_path = args.data_dir

   # Construct variables
   wavX = '.wav'
   meldict = {}

   # Get filenames in dataset
   filenames = os.listdir(master_path)
   filepaths = [os.path.join(master_path, f) for f in filenames if f.endswith(wavX)]

   
