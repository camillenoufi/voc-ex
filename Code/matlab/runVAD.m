% perform vad on balanced dataset

path = '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/audio/wav/dev/';


load('/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/metadata/mat_files/split/local_train_bal.mat');
%%
train_labels = unique([local_train_bal.localeNum]);
dev_labels = [dev_dir.localeNum];
ix = ismember(dev_labels,train_labels);
dev_top = dev_dir(ix==1);
[yl,yi] = datasample([dev_top.localeNum],50);
hist(yl,unique(yl));
%%
local_dev = dev_top(yi);
local_dir = fullfile(path,'local_dev');
if ~exist(local_dir,'dir')
    mkdir local_dir;
end

%%
addpath('sap-voicebox-master/voicebox');
for i=1:length(local_dev)
    fname = local_dev(i).name;
    [y,fs] = audioread(fullfile(path,fname));
    [vs,zo] = v_vadsohn(y,fs);
    len = min(length(y),length(vs));
    out = y(1:len);
    out = out(vs==1);
    audiowrite(fullfile(local_dir,fname),y,fs);
    disp(i);
end