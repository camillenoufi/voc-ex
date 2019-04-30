# shell commands to run variants of the Accent Detection NN script: experiments.py

# command options can be found by typing python experiments.py -h

# Within the Azure backup directory, run the CNN model test script using the GPU
#python experiments.py --dir /usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/azure-backup/dataset --model cnn  --gpu_flag 1 --test_flag 1

# Within the Azure backup directory, train the CNN model using the GPU
python experiments.py --dir /usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/azure-backup/dataset --model cnn  --gpu_flag 1 --test_flag 0
