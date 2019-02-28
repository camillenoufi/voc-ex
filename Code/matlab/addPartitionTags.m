% add partition label to full entries structure
% 1 - train
% 2 - dev
% 3 - test

%load('/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/metadata/entry_data_structures/final_valid_entries.mat');
n = length(entries);
path = '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/audio/wav';
test_path = fullfile(path,'test');
dev_path = fullfile(path,'dev');
train_path = fullfile(path,'train');
notfoundids = [];

%%
for i=1:n
    fname = entries(i).name;
    %in train set
    if(exist(fullfile(train_path,fname),'file'))
        entries(i).folder = train_path;
        entries(i).set = 1;
    %in dev set
    elseif(exist(fullfile(dev_path,fname),'file'))
        entries(i).folder = dev_path;
        entries(i).set = 2;
    %in test set
    elseif(exist(fullfile(test_path,fname),'file'))
        entries(i).folder = test_path;
        entries(i).set = 3;
    else
        notfoundids = [notfoundids; i];
    end
end
        
        
        
        