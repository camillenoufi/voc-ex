convertStructarr2csv(dev_entries_top10, '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/metadata/dev_labels.csv');
convertStructarr2csv(test_entries_top10, '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/metadata/test_labels.csv');
convertStructarr2csv(train_entries_top10_bal, '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/metadata/trainBalanced_labels.csv');
convertStructarr2csv(train_entries_top10, '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/metadata/trainAll_labels.csv');


function convertStructarr2csv(this_struct,fout)
writetable(cell2table(struct2cell(this_struct)','VariableNames',fieldnames(this_struct)),fout);
end

