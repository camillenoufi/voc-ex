# Within the Azure backup directory, train the CRNN model using the GPU and the STFT spectrograms
python experiments.py --dir /usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/azure-backup/dataset --model nolstm  --gpu_flag 1 --test_flag 1 --feature 0 --model_file crnn_state_dict-0-10-32-0.3.pt 
