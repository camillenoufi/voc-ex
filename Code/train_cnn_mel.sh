# Within the Azure backup directory, train the CRNN model using the GPU and the STFT spectrograms
python experiments.py --dir /usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/azure-backup/dataset --model cnn  --gpu_flag 1 --test_flag 0 --feature 0
