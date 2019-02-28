% check if final metadata structure and audio directory files all match

wavpath = '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/audio/wav';
extStr = '.wav';
issueIds2 = [];

for i=1:length(entries)
    [~,fname,ext] = fileparts(entries(i).name);
    entries(i).name = [fname extStr];
    entries(i).folder = wavpath;
    if ~exist(fullfile(wavpath,entries(i).name),'file')
        issueIds2 = [issueIds2; i];
    end
end