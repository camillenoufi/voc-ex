%%%% Partition full Dataset (technically refined)
% Select Top 10 countries
% Balance Dataset
% Partition into train and test


%% Select Top 10 countries from complete (technically refined) dataset

%Params:
C = 10; % number of countries to subselect

% Data: (this particular import will be called 'entries')
importfile('/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/metadata/mat_files/full/final_valid_entries.mat');

% Get country-locale numeric codes for C most common in training set
localeLabels = [entries.localeNum];
[nOcc,uLbl] = hist(localeLabels,unique(localeLabels));
[nOcc,i] = sort(nOcc,'descend');
uLbl = uLbl(i);
uLbl = uLbl(1:C);
nOcc = nOcc(1:C);

% extract  entries that are part of this set
ix = ismember(localeLabels,uLbl);
entries_top10 = entries(ix);

%% Balance Dataset

entries_top10bal = [];
N = nOcc(end-1); %class size will be not the lowest occurence number, but the second lowest
for i=1:C
    this_entries = entries_top10([entries_top10.localeNum] == uLbl(i));
    M = length(this_entries);
    if M > N
        subset = this_entries(randsample(M,N),:);
        entries_top10bal = [entries_top10bal; subset];
    else
        entries_top10bal = [entries_top10bal; this_entries];
    end
end

%% Partition into Train / Test

pTrain = .9;
M = length(entries_top10bal);

rng(1);
train_idx = randperm(M,round(pTrain*M));
[~,test_idx] = setdiff((1:M),train_idx);

entries_train = entries_top10bal(train_idx);
entries_test =  entries_top10bal(test_idx);

outpath = '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/azure-backup/dataset';
trainDir = 'train';
testDir = 'test';

addpath('./struct2csv');
structarr2csv(entries_train,fullfile(outpath,trainDir,'train_labels.csv'))
structarr2csv(entries_test,fullfile(outpath,testDir,'test_labels.csv'))

%% Perform VAD and move selected audio into partition location folder
inpath = '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/audio/wav/all';

% addpath('./sap-voicebox-master/voicebox');
% performVAD(entries_train,inpath,outpath,trainDir);
% performVAD(entries_test,inpath,outpath,testDir);

selectAudio(entries_train,inpath,outpath,trainDir);
selectAudio(entries_test,inpath,outpath,testDir);

%% Helper Funcs

function importfile(fileToRead1)

% Import the file
newData1 = load('-mat', fileToRead1);

% Create new variables in the base workspace from those fields.
vars = fieldnames(newData1);
for i = 1:length(vars)
    assignin('base', vars{i}, newData1.(vars{i}));
end

end

function performVAD(theStruct,in_path,out_path,partition)
    
    disp(partition);
    
    onset = 9.6; %75 bpm, downbeat of 5th measure
    off = 22.4; %after "wretch"
    out_path = fullfile(out_path,partition);
    
    if ~exist(out_path,'dir')
        mkdir(out_path);
    end

    for i=1:length(theStruct)
        fname = theStruct(i).name;
        [y,fs] = audioread(fullfile(in_path,fname));
        y = y(onset*fs:off*fs); %remove intro
%         [vs,~] = v_vadsohn(y,fs);
%         len = min(length(y),length(vs));
%         y_out = y(1:len);
%         y_out = y_out(vs==1);
        audiowrite(fullfile(out_path,fname),y,fs);
        disp(i);
    end
end

function selectAudio(theStruct,in_path,out_path,partition)
    
    disp(partition);
    
    onset = 9.6; %75 bpm
    offset = 22.4; 
    out_path = fullfile(out_path,partition);
    
    if ~exist(out_path,'dir')
        mkdir(out_path);
    end

    for i=1:length(theStruct)
        fname = theStruct(i).name;
        [y,fs] = audioread(fullfile(in_path,fname));
        y = y(round(onset*fs):round(offset*fs)); %remove intro
%         [vs,~] = v_vadsohn(y,fs);
%         len = min(length(y),length(vs));
%         y_out = y(1:len);
%         y_out = y_out(vs==1);
        audiowrite(fullfile(out_path,fname),y,fs);
        disp(i);
    end
end