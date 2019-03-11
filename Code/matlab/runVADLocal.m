% perform vad on balanced dataset

addpath('sap-voicebox-master/voicebox');

path = '/Users/camillenoufi/Documents/datasets/VocEx-local/local_balanced/';

files = dir(fullfile(path,'*.wav'));
for i=1:length(files)
    [y,fs] = audioread(fullfile(files(i).folder,files(i).name));
    [vs,zo] = v_vadsohn(y,fs);
    len = min(length(y),length(vs));
    out = y(1:len);
    out = out(vs==1);
    audiowrite(fullfile(path,files(i).name),out,fs);
    disp(i);
end

path = '/Users/camillenoufi/Documents/datasets/VocEx-local/local_dev/';

%%
files = dir(fullfile(path,'*.wav'));
for i=1:length(files)
    [y,fs] = audioread(fullfile(files(i).folder,files(i).name));
    [vs,zo] = v_vadsohn(y,fs);
    len = min(length(y),length(vs));
    out = y(1:len);
    out = out(vs==1);
    audiowrite(fullfile(path,files(i).name),out,fs);
    disp(i);
end