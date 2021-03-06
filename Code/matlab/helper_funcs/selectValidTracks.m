function entries = readInFiles(root_path)
% returns 'entries' struct


%get all files
savepath = fullfile(root_path,'metadata/entry_metadata_structs','full_entries.mat');
if 0 %exist(savepath, 'file')
    tmp = load(savepath); %Creates new directory within audio directory for evaluation snippets
    entries = tmp.entries;
    clear tmp;
else
    entries = dir(fullfile(root_path,'*.m4a'));
end

% Data specifications
metadata = tdfread(fullfile(root_path,'amazing_grace.tsv'));
noisedata = readtable(fullfile(root_path,'BackgroundNoiseCheckComplete.txt'),...
    'Delimiter',' ','ReadVariableNames',false);
num_entries = length(entries);
valid_fs = 22050;
min_len = 4e6;
max_NSR = 0.05;

% Audio write specifications
trimmed_audio_dir = fullfile(root_path,'eval_audio_trimmed');
if ~exist(trimmed_audio_dir, 'dir')
    mkdir(trimmed_audio_dir); %Creates new directory within audio directory for evaluation snippets
end
trim_start = 23;
trim_end = 37.5;
fade_durations = [ 250 200 ];       % fade-in and fade-out durations (ms)
fade_windows = { @(N)(hanning(N).^2) @(N)(hanning(N).^2) }; 

disp('reading in each file...');


tic;
%construct data structure
for i=1:num_entries
    disp(i);
    % extract audio
    [~,fname,~] = fileparts(entries(i).name); %for metadata table
    savename = fullfile(trimmed_audio_dir,strcat(fname,'_eval.wav'));
    try 
        [y, fs] = audioread(fullfile(entries(i).folder,entries(i).name));
        track_len = size(y,1);
        if (fs == valid_fs) && (track_len > min_len)
            fmatch = strcat(fname,'.wav'); %for noisedata table 
            this_noisedata = noisedata(strcmp(noisedata.Var1,fmatch),:);
            NSR_str = this_noisedata{1,'Var6'};
            NSR = str2double(NSR_str{1,1});
            if (NSR < max_NSR)
                %metadata
                m = find(strcmp(cellstr(metadata.performance_id),fname));
                entries(i).account_id = metadata.account_id(m);
                entries(i).performance_id = metadata.performance_id(m,:);
                entries(i).fs = fs;
                entries(i).samp_len = track_len;
                entries(i).ratio = NSR;
                entries(i).birth_year = str2double(metadata.birth_year(m,:));
                entries(i).gender = metadata.gender(m,1);
                entries(i).device_os = metadata.device_os(m,:);
                entries(i).headphones = str2double(metadata.headphones(m,:));
                entries(i).city_id = str2double(metadata.city_id(m,:));
                entries(i).locale = metadata.locale(m,:);
                entries(i).country = metadata.country(m,:);
                entries(i).latitude = str2double(metadata.latitude(m,:));
                entries(i).longitude = str2double(metadata.longitude(m,:));
                entries(i).creation_timestamp = metadata.creation_timestamp(m);
                % noise data
                entries(i).perf = this_noisedata{1,'Var1'}{1};
                entries(i).negpeak = this_noisedata{1,'Var2'};
                entries(i).pospeak = this_noisedata{1,'Var3'};
                entries(i).intropow = this_noisedata{1,'Var4'};
                entries(i).restpow = this_noisedata{1,'Var5'};
                entries(i).clean = this_noisedata{1,'Var8'}{1};
                entries(i).trash = this_noisedata{1,'Var9'};
                entries(i).nfiles = this_noisedata{1,'Var10'};
                entries(i).nclean = this_noisedata{1,'Var11'};
                %write evaluation snippet to file
                try 
                    samples = y((fs*trim_start):(fs*trim_end));
                    faded = fade( samples, fs, fade_durations, fade_windows );
                    audiowrite(savename,faded,fs);
                    entries(i).evalTrim_exists = 1;
                catch
                    entries(i).evalTrim_exists = 0;
                end
            else
                entries(i).fs = fs;
                entries(i).ratio = NSR;
            end
        else
            entries(i).fs = fs;
        end
    catch
        entries(i).fs = -1;
    end
    save(savepath,'entries');
end

%keep only completed data entries with valid metadata
entries = entries(1:num_entries);
clear metadata;
clear noisedata;

%remove unneeded fields
entries = rmfield(entries,'isdir');
entries = rmfield(entries,'date');
entries = rmfield(entries,'datenum');


%% Remove invalid files

invalid_entries_idx = find([entries.fs] ~= valid_fs);
entries(invalid_entries_idx) = [];
%noisy_idx = entries(high_NSR_idx); %save

toc;
disp('File read complete');
disp('Saving completed valid_entries.mat to disk...');
save(savepath,'entries');
disp('Save complete');

end



