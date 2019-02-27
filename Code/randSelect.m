cd '/usr/ccrvma/media/projects/jordan/Experiments/VocEx1.1/Code';
addpath('./helper_funcs/')

files = dir(fullfile('/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/eval_audio_trimmed/mp3','*.mp3'));
%%
subset = struct2table(files(randperm(length(files),3000)));
subset = subset(:,1);
%%
writetable(subset,'/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/eval_audio_trimmed/mp3/subsetList_3000.csv');