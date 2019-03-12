% must have full metadata .mat files for all 3 partitions loaded

% Get country-locale numeric codes for 10 most common in training set
C = 10;
train_locale_labels = [train_entries.localeNum];
[nOcc,uLbl] = hist(train_locale_labels,unique(train_locale_labels));
[nOcc,i] = sort(nOcc,'descend');
uLbl = uLbl(i);
uLbl = uLbl(1:C);
nOcc = nOcc(1:C);

% extract train entries that are part of this set
ix = ismember(train_locale_labels,uLbl);
train_entries_top10 = train_entries(ix);


% extract dev entries that are part of this set
dev_locale_labels = [dev_entries.localeNum];
ix = ismember(dev_locale_labels,uLbl);
dev_entries_top10 = dev_entries(ix);

% extract test entries that are part of this set
test_locale_labels = [test_entries.localeNum];
ix = ismember(test_locale_labels,uLbl);
test_entries_top10 = test_entries(ix);

%create a balanced version of this train set via undersampling
train_entries_top10_bal = [];
rng('default'); % For reproducibility
N =  median(nOcc);
for i=1:C
    nEntries = train_entries_top10([train_entries_top10.localeNum] == uLbl(i));
    nOcc = length(nEntries);
    if nOcc > N
        subset = nEntries(randsample(length(nEntries),N),:);
        train_entries_top10_bal = [train_entries_top10_bal; subset];
    else
        train_entries_top10_bal = [train_entries_top10_bal; nEntries];
    end
end











