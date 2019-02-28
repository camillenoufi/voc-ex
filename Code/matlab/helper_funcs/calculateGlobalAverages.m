function [ global_rms, global_drms, global_psd, global_pkpsd ] = calculateGlobalAverages( entries, lim )
%% global smoothed RMS

lim = length(entries);

rms_length = length(entries(1).smoothrms_mm12);
global_rms = zeros(rms_length,1);
for i= 1:rms_length
    csum_f = 0;
    for j = 1:lim 
        csum_f = csum_f + entries(j).smoothrms_mm12(i);
    end
    global_rms(i) = csum_f / lim;
end

%% global smoothed dRMS

drms_length = length(entries(1).dsmoothrms_mm12);
global_drms = zeros(drms_length,1);
for i= 1:drms_length 
    csum_f = 0;
    for j = 1:lim 
        csum_f = csum_f + entries(j).dsmoothrms_mm12(i);
    end
    global_drms(i) = csum_f / lim;
end

%% global PSD

pk_length = length(entries(1).psd_mm12);
global_psd = zeros(pk_length, 2);
for i= 1: pk_length 
    csum_f = 0;
    for j = 1: lim 
        csum_f = csum_f + entries(j).psd_mm12(i,2);
    end
    global_psd(i,2) = csum_f / lim;
end
global_psd(:,1) = entries(1).psd_mm12(:,1);

%% global PSD peaks

pk_length = length(entries(1).psdPks_mm12);
global_pkpsd = zeros(pk_length, 2);
for i= 1: pk_length 
    csum_f = 0;
    csum_p = 0;
    for j = 1: lim 
        csum_f = csum_f + entries(j).psdPks_mm12(i,1);
        csum_p = csum_p + entries(j).psdPks_mm12(i,2);
    end
    global_pkpsd(i,1) = csum_f / lim;
    global_pkpsd(i,2) = csum_p / lim;
end 

end

