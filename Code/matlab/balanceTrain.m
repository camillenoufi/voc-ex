all_locale_labels = [train_entries.localeNum];

[nOcc,uLbl] = hist(all_locale_labels,unique(all_locale_labels));
[nOcc,i] = sort(nOcc,'descend');
uLbl = uLbl(i);
uLbl = uLbl(1:10);





