% tag countries with number labels

n = length(entries);

cn = 0;
crStr = entries(1).country;

for i=1:n
   if(strcmp(entries(i).country,crStr))
       entries(i).countryNum = cn;
   else
       crStr = entries(i).country;
       cn = cn+1;
       entries(i).countryNum = cn;
   end
end