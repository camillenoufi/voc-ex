% select idx of balanced top 10 locales in train dataset

n = length(top_locale_arr);
cl = -1;
top_locale_arr = sort(top_locale_arr,'ascend');
subset_ids = [];

for i=1:n
    if(top_locale_arr(i) ~= cl)
        cl = top_locale_arr(i);
        subset_ids = [subset_ids, i:i+9];
    end
end
        

