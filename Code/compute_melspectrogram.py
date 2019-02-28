"""
Compute the mel-spectrogram for batch of audio files
"""

import argparse
import random
import os
import librosa

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/audio/wav/train/local_balanced', help="Directory with the master valid DAMP-VocEx dataset")


if __name__ == '__main__':
   
   args = parser.parse_args()
   
   assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
   
   master_path = args.data_dir
   
   # Get filenames in master directory and shuffle randomly
   filenames = os.listdir(master_path)
   
   