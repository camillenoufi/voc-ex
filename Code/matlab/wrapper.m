%% Amazing Grace (DAMP) data processing wrapper %%
% Camille Noufi - November 2018
%
% Processing Steps:
% 1) Read in files and extract audio samples
% 2) Extract samples pertaining to intervals of interest
% 3) Create STFT for each interval sample
% 4) Get data describing pitch contour
% 5) Get RMS Info
% 6) Calculate Correlation measures between RMS and pitch contour

%% 0) SETUP
cd '/usr/ccrma/media/projects/jordan/Experiments/VocEx1.1/Code';
addpath('./helper_funcs/')
addpath('./sap-voicebox-master/voicebox');
addpath('./nestedSortStruct-master');

%%  1) Read in files and extract audio samples
% returns 'entries' struct with fields:
% - name (filename.ext)
% - sampling rate (fs)
% - several fields of metadata (see function for names of fields)

disp('Reading in audio and metadata....');

root_path = '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG';
entries = readInFiles(root_path); %filepath currently described within function

%% extract specific demographic (optional)
idx = [];

% for i=1:length(entries_lim)
%     if (strncmp(entries_lim(i).country,'GB',2))
%         idx = [idx, i];
%     end
% end

idx = find([entries_lim.birth_year] <= 1988);
entries = entries_lim(idx);
nO = length(entries);

%% 2) Extract samples pertaining to intervals of interest
% mm 12, 27, (43, 59).  All are assumed to be ascending 4ths

mm12_start = 23.3; 
mm12_end = 27.8;

% 3) Calculate Overall Power Features and Store in main struct

windowLength = 1024; overlapLength = windowLength/2; filterLength = 10; nPSDpeaks = 10; 
%mm12; 

% currently only calculating spectrogram
entries = calculatePowerFeatures(entries,lim,mm12_start,mm12_end,windowLength,overlapLength,filterLength, nPSDpeaks);
%%
lim = length(entries);
T = entries(1).T;
F = entries(1).F;
B = zeros(size(entries(1).B,1),size(entries(1).B,2));
for i = 1:lim
    B = B + entries(i).B;
end
B = B / lim;
%%
Bplot = 10*log10(B');
figure; imagesc(T+mm12_start,F(1:128),Bplot(1:128,:));
axis xy; colorbar;
title('Interval Spectral Contents: Global Average'); xlabel('Time (s)'); ylabel('Frequency (Hz)');

%%
% 4) Calculate Global Averages
[ global_rms, global_drms, global_psd, global_pkpsd ] = calculateGlobalAverages( entries, lim );

figure('NumberTitle', 'off', 'Name', 'Global Average Power of Interval (mm12)');

subplot(2,2,1);
plot((1:length(global_rms))*windowLength/2/fs, global_rms);
title('Global Average RMS Power'); xlabel('Time'); ylabel('Amplitude');


subplot(2,2,3);
plot((1:length(global_rms)-1)*windowLength/2/fs,global_drms);
title('Global Average \DeltaRMS'); xlabel('Time'); ylabel('Amplitude');

subplot(2,2,2);
plot(global_psd(1:100,1),global_psd(1:100,2));
title('Global Average Power Spectral Density'); xlabel('Freq. (Hz)'); ylabel('Amplitude');

subplot(2,2,4);
plot(global_pkpsd(:,1),global_pkpsd(:,2),'x');
title('Global Average of 10 Max Power Peaks'); xlabel('Freq. (Hz)'); ylabel('Amplitude');

% 'like'
% mm12 like start: 23.3 23.7
% like end: 24.3 24.0

starts = 23.7;
ends = 23.9;
windowLength = 1024; overlapLength = windowLength/2; filterLength = 1; nPSDpeaks = 6; 
entries = calculatePowerFeatures(entries,lim,starts,ends,windowLength,overlapLength,filterLength, nPSDpeaks);
[ global_rms, global_drms, global_psd, global_pkpsd ] = calculateGlobalAverages( entries, lim );

figure('NumberTitle', 'off', 'Name', 'Global Average Power of Pitch 1');

subplot(2,2,1);
plot((1:length(global_rms))*windowLength/2/fs, global_rms);
title('Global Average RMS Power'); xlabel('Time'); ylabel('Amplitude');


subplot(2,2,3);
plot((1:length(global_rms)-1)*windowLength/2/fs,global_drms);
title('Global Average \DeltaRMS'); xlabel('Time'); ylabel('Amplitude');

subplot(2,2,2);
plot(global_psd(1:200,1),global_psd(1:200,2));
title('Global Average Power Spectral Density'); xlabel('Freq. (Hz)'); ylabel('Amplitude');

subplot(2,2,4);
plot(global_pkpsd(:,1),global_pkpsd(:,2),'x');
title('Global Average of 10 Max Power Peaks'); xlabel('Freq. (Hz)'); ylabel('Amplitude');

like_rms = global_rms;
like_psd = global_psd;
% 'me'
% me start: 24.3 26.0
% me end: 27.8 26.3

starts = 25.1;
ends = 25.3;
windowLength = 1024; overlapLength = windowLength/2; filterLength = 1; nPSDpeaks = 6; 
entries = calculatePowerFeatures(entries,lim,starts,ends,windowLength,overlapLength,filterLength, nPSDpeaks);
[ global_rms, global_drms, global_psd, global_pkpsd ] = calculateGlobalAverages( entries, lim );

figure('NumberTitle', 'off', 'Name', 'Global Average Power of Pitch 2');

subplot(2,2,1);
plot((1:length(global_rms))*windowLength/2/fs, global_rms);
title('Global Average RMS Power'); xlabel('Time'); ylabel('Amplitude');


subplot(2,2,3);
plot((1:length(global_rms)-1)*windowLength/2/fs,global_drms);
title('Global Average \DeltaRMS'); xlabel('Time'); ylabel('Amplitude');

subplot(2,2,2);
plot(global_psd(1:200,1),global_psd(1:200,2));
title('Global Average Power Spectral Density'); xlabel('Freq. (Hz)'); ylabel('Amplitude');

subplot(2,2,4);
plot(global_pkpsd(:,1),global_pkpsd(:,2),'x');
title('Global Average of 10 Max Power Peaks'); xlabel('Freq. (Hz)'); ylabel('Amplitude');

me_rms = global_rms;
me_psd = global_psd;
figure('NumberTitle', 'off', 'Name', 'RMS Pitch Comparison');
plot((1:length(like_rms))*windowLength/2/fs,like_rms_all); hold on; plot((1:length(me_rms))*windowLength/2/fs,me_rms_all);
legend('Pitch 1 (Like)', 'Pitch 2 (Me)');
title('RMS Comparison of Pitch Segments'); xlabel('Time (s)'); ylabel('Amplitude');
