function [ entries ] = calculatePowerFeatures(entries, lim, sec_start, sec_end, windowLength, overlapLength,filterLength, np)

lim = length(entries);

% don't zeropad
zeropad = 0;
% create moving average filter for rms
maf = ones(1, filterLength)/filterLength;


for i=1:lim
    % extract data at interval location
    X = audioread(fullfile(entries(i).folder,entries(i).name));
    fs = entries(i).fs;
    X = X(sec_start*fs:sec_end*fs);
    % Remove DC component
    X = X-mean(X);
    %normalize
    X = X/max(abs(X));
    
    % calculate windowed RMS
    entries(i).rms_mm12 = rms(X, windowLength, overlapLength, zeropad);
    % calculate the smoothed RMS (via moving average filter)
    entries(i).smoothrms_mm12 = filter(maf, 1, entries(i).rms_mm12);
    % calculate the derivative of the smoothed-RMS
    entries(i).dsmoothrms_mm12 = diff(entries(i).smoothrms_mm12);
    %calculate Welch PSD
    [psd,F] = pwelch(X,windowLength,[],[],fs); % 50% overlap by default
    entries(i).psd_mm12 = [F, psd];
    %extract top peaks + their frequency locations
    [pks,locs] = findpeaks(psd,'NPeaks',np, 'SortStr','descend');
    locs = (fs/windowLength)*(locs-1);
    maxpks = [locs,pks];
    entries(i).psdPks_mm12 = sortrows(maxpks,1);
    
    [T,F,B] = v_spgrambw(X,fs, 'pJcw');
    entries(i).T = T;
    entries(i).F = F;
    entries(i).B = B;
    
end


end

