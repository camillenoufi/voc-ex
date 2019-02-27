cd '/usr/ccrma/media/projects/jordan/Experiments/VocEx1.1/Code';
addpath('./helper_funcs/')

files = dir(fullfile('/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/wav_top/wav/d001','*.csv'));
%%

path = fullfile(files(1).folder,files(1).name);
sum_pitchtrack = csvread(path,1,0);
len = size(sum_pitchtrack,1);
sum_pitchtrack = zeros(len,3);
sum_pc = zeros(len,1);
sum_pc_cents = sum_pc;
%%
for i=1:length(files)
    path = fullfile(files(i).folder,files(i).name);
    this_pitchtrack = csvread(path,1,0);
    this_len = size(this_pitchtrack,1);
    f = this_pitchtrack(:,2);
    m = (12*log2(f/440) + 69) - 3; %adjust to Eb being 0 in PC
    %pc = mod(round(m),12);
    pc_cents = mod(m,12);
    if this_len<len
        len = this_len;
        sum_pitchtrack = sum_pitchtrack(1:len,:) + this_pitchtrack;
        %sum_pc = sum_pc(1:len) + pc;
        sum_pc_cents = sum_pc_cents(1:len) + pc_cents;
    else
        sum_pitchtrack = sum_pitchtrack + this_pitchtrack(1:len,:);
        %sum_pc = sum_pc + pc(1:len);
        sum_pc_cents = sum_pc_cents + pc_cents(1:len);
    end
end
%%
sum_pitchtrack = [sum_pitchtrack, sum_pc_cents];
sum_pitchtrack = sum_pitchtrack/i;
%%
sum_pitchtrack = [sum_pitchtrack, round(sum_pitchtrack(:,4))];

figure; plot(sum_pitchtrack(range,1),sum_pitchtrack(range,4)); hold on; plot(sum_pitchtrack(range,1),sum_pitchtrack(range,5)); legend('PC','PC + cents');