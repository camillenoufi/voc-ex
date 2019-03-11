function structarr2csv(this_struct,fout)
writetable(cell2table(struct2cell(this_struct)','VariableNames',fieldnames(this_struct)),fout);
end

