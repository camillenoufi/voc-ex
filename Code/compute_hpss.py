"""
Compute the hpss-spectrogram for batch of audio files
"""

import argparse
import os
import numpy as np, scipy as sp
import librosa
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='/Users/camillenoufi/Documents/datasets/VocEx-local/local_dev/', help="Directory with the master valid DAMP-VocEx dataset")
parser.add_argument('--out_file', default='dict_dev_feats_NotNorm.pkl', help="output Pickle File containing feature dictionary")


# chop spectrogram into feature slices
def sliceFeatures(S,nsec):
    fsec = .60 #feature length in seconds, 1 beat
    feature_list = []
    (_,N) = S.shape
    slice_size = int(np.floor((N/nsec)*fsec)) #frame/sec * feature length in seconds
    hop_size = int(np.floor(slice_size*0.50))

    i=0;
    while i < N:
        this_slice = S[:,i:i+slice_size]
        feature_list.append(this_slice)
        i = i+hop_size
    return feature_list

# Compute STFT features and slice
def computeSTFTSlices(file):
    x, fs = librosa.load(os.path.join(master_path, f))
    xlen = len(x) #length of audio file in samples
    nsec = xlen/fs #length of audio file in seconds

    S = librosa.stft(x, n_fft=1024, hop_length=256)
    H, P = librosa.decompose.hpss(S)
    
    H = adjustScaling(H)
    P = adjustScaling(P)
    
    H_feature_list = sliceFeatures(H,nsec)
    P_feature_list = sliceFeatures(P,nsec)
    return (H_feature_list, P_feature_list)

def adjustScaling(S):
    S = librosa.amplitude_to_db(np.abs(S)) #convert to log spectrogram and drop phase
    nfbins = int(np.floor(S.shape[0]/2))
    S = S[:nfbins,:] # lower half of the frequency bins selected
    return S

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

   # for a file, compute stft and slice it into smaller features, then add array of features to the dictionary
   i=1
   for f in filepaths:
       feature_list_tupleHP = computeSTFTSlices(f)
       dict_file2feature_list[f] = feature_list_tupleHP
       print(i)
       i=i+1

   #save dictionary to file
   out_path = os.path.join(master_path, out_file)
   fout = open(out_path,"wb")
   pickle.dump(dict_file2feature_list,fout)
   fout.close()
   print('features successfully saved to file at: ' + out_path)
