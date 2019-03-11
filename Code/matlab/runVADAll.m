% perform vad on balanced dataset

addpath('sap-voicebox-master/voicebox');

metadata_dirpath = '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/metadata/mat_files/split';
audio_path = '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/audio/wav/';

% Metadata filenames
trainBal_metadata = 'train_top10localesBalanced_metadata.mat';
trainAll_metadata = 'train_top10locales_metadata.mat';
dev_metadata = 'dev_top10locales_metadata.mat';
test_metadata = 'test_top10locales_metadata.mat';

% Partition subdirectories
train_dir = 'train';
dev_dir = 'dev';
test_dir = 'test';

%% Setup Specific VAD call
in_path = fullfile(audio_path,test_dir);
out_dir = 'VAD_top10_all';
metadata_filepath = fullfile(metadata_dirpath,test_metadata);

performVAD(metadata_filepath,in_path,out_dir);

disp('VAD complete.  Files at: ');
disp(fullfile(in_path,out_dir));



%% VAD Call Function
function performVAD(metadata_path, in_path,out_dir)

metadata = load(metadata_path);
fieldname = fieldnames(metadata);
metadata = metadata.(fieldname{1});


out_path = fullfile(in_path,out_dir);
if ~exist(out_path,'dir')
    mkdir(out_path);
end

for i=1:length(metadata)
    fname = metadata(i).name;
    [y,fs] = audioread(fullfile(in_path,fname));
    [vs,~] = v_vadsohn(y,fs);
    len = min(length(y),length(vs));
    y_out = y(1:len);
    y_out = y_out(vs==1);
    audiowrite(fullfile(out_path,fname),y_out,fs);
    disp(i);
end

end