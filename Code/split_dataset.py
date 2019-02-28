"""
Split the valid tracks of the DAMP dataset into test, train and dev subdirectories
"""

import argparse
import random
import os
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/audio/wav/', help="Directory with the master valid DAMP-VocEx dataset")
#parser.add_argument('--output_dir', default='data/64x64_SIGNS', help="Where to write the new data")


if __name__ == '__main__':
   
   args = parser.parse_args()
   
   assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
   
   master_path = args.data_dir 
   
   # Define the data directories
   train_data_dir = os.path.join(args.data_dir, 'train')
   test_data_dir = os.path.join(args.data_dir, 'test')
   dev_data_dir = os.path.join(args.data_dir, 'dev')
   
   # Get filenames in master directory
   filenames = os.listdir(master_path)
   print(filenames)
   print(aaaa)
   #filenames = [os.path.join(master_path, f) for f in filenames if f.endswith('.wav')]
   
   # Split audio files master_path into 80% train, 10% dev, and 10% test
   random.seed(224)
   filenames.sort()
   random.shuffle(filenames)
   
   split1 = int(0.8 * len(filenames))
   split2 = int(0.9 * len(filenames))
   train_filenames = filenames[:split1]
   dev_filenames = filenames[split1:split2]
   test_filenames = filenames[split2:]
   
   split_filenames = {'train': train_filenames, 'dev': dev_filenames, 'test': test_filenames}
   split_list = ['train','dev','test']
   
   
   
   for path in split_list:
       shutil.move()
   
   
   
   
   



