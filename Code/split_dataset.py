"""
Split the valid tracks of the DAMP dataset into test, train and dev subdirectories
"""

import argparse
import random
import os
import copy


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/audio/wav/', help="Directory with the master valid DAMP-VocEx dataset")
#parser.add_argument('--output_dir', default='data/64x64_SIGNS', help="Where to write the new data")


# Create Split Partition Directory
def createPartitionDir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    else:
        print("Warning: dir {} already exists".format(dirpath))
        

# Main Routine
if __name__ == '__main__':
   
   args = parser.parse_args()
   
   assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
   
   master_path = args.data_dir 
   
   # Define file type
   f_ext = '.wav'
   
   # Define the data directories
   train_data_dir = os.path.join(args.data_dir, 'train')
   createPartitionDir(train_data_dir)
   
   test_data_dir = os.path.join(args.data_dir, 'test')
   createPartitionDir(test_data_dir)
   
   dev_data_dir = os.path.join(args.data_dir, 'dev')
   createPartitionDir(dev_data_dir)
   
   # Get filenames in master directory and shuffle randomly
   filenames = os.listdir(master_path)
   random.seed(224)
   filenames.sort()
   random.shuffle(filenames)
   
   # Create filepath lists of master and partitions
   split_filepaths = copy.deepcopy(filenames)
   master_filepaths = [os.path.join(master_path, f) for f in filenames if f.endswith(f_ext)]

   # Split audio files in master_path into 80% train, 10% dev, and 10% test 
   split1 = int(0.8 * len(filenames))
   split2 = int(0.9 * len(filenames))
   
   split_filepaths[:split1] = [os.path.join(train_data_dir, f) for f in filenames[:split1] if f.endswith(f_ext)]
   split_filepaths[split1:split2] = [os.path.join(dev_data_dir, f) for f in filenames[split1:split2] if f.endswith(f_ext)]
   split_filepaths[split2:] = [os.path.join(test_data_dir, f) for f in filenames[split2:] if f.endswith(f_ext)]
   
   
   # Move files into their correct partition folder
   for m, p in zip(master_filepaths, split_filepaths):
       os.rename(m,p)
       
   
   
   
   
   



