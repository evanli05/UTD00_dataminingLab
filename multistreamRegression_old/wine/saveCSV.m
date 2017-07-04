src = srcData.input';
src(:,12) = srcData.output';


tar = tarData.input';
tar(:,12) = tarData.output';


csvwrite('source_wine.csv',src);
csvwrite('target_wine.csv',tar);
