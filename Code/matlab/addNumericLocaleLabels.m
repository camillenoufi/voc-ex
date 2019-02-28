% tag countries with number labels

[~,index] = sortrows({entries.locale}.'); entries = entries(index); clear index
n = length(entries);
ln = 0;
lStr = entries(1).locale;

for i=1:n
   if(strcmp(entries(i).locale,lStr))
       entries(i).localeNum = ln;
   else
       lStr = entries(i).locale;
       ln = ln+1;
       entries(i).localeNum = ln;
   end
end