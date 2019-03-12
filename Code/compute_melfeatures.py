"""
Compute the mel-spectrogram for batch of audio files
"""

import argparse
import os
import numpy as np, scipy as sp
import librosa, librosa.display
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='/Users/camillenoufi/Documents/datasets/VocEx-local/local_dev/', help="Directory with the master valid DAMP-VocEx dataset")
parser.add_argument('--out_file', default='dict_dev_feats_NotNorm.pkl', help="output Pickle File containing feature dictionary")


# chop spectrogram into feature slices
def sliceFeatures(S,nsec):
    feature_list = []
    (_,N) = S.shape
    slice_size = int(np.floor((N/nsec)*4.75)) #frame/sec * feature length in seconds
    hop_size = int(np.floor(slice_size*0.50))

    i=0;
    while i < N:
        this_slice = S[:,i:i+slice_size]
        feature_list.append(this_slice)
        i = i+hop_size
    return feature_list

# Compute Mel spectrogram features and slice
def computeMelFeatureSlices(file):
    x, fs = librosa.load(os.path.join(master_path, f))
    xlen = len(x)
    nsec = xlen/fs
    S = librosa.feature.melspectrogram(x, sr=fs, n_fft=2048, hop_length=512, power=2.0)
    #S -= (np.mean(S, axis=0) + 1e-8)
    S = librosa.power_to_db(S)
    S = S[:80,:]
    ids = np.where(S<-60)
    S[ids[0],ids[1]] = 0
    #S = sp.stats.zscore(S,axis=None);

    feature_list = sliceFeatures(S,nsec)
    return feature_list



if __name__ == '__main__':

    # Process command line
   args = parser.parse_args()
   assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)
   master_path = args.data_dir

   print('Using dataset at: ')
   print(master_path)
   out_file = args.out_file
   print('Pickle (.pkl) Feature File will save at: ')
   print(out_file)

   # Declarations
   wavX = '.wav'
   dict_file2feature_list = {}

   # Get filenames in dataset
   filenames = os.listdir(master_path)
   filepaths = [os.path.join(master_path, f) for f in filenames if f.endswith(wavX)]
   filepaths.sort()

   # for a file, compute mel spectrogram and slice it into smaller features, then add array of features to the dictionary
   i=1
   for f in filepaths:
       feature_list = computeMelFeatureSlices(f);
       dict_file2feature_list[f] = feature_list
       print(i)
       i=i+1

   #save dictionary to file
   out_path = os.path.join(master_path, out_file)
   fout = open(out_path,"wb")
   pickle.dump(dict_file2feature_list,fout)
   fout.close()
   print('features successfully saved to file at: ' + out_path)
