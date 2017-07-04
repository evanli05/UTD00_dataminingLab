src = srcData.input';
src(:,17) = srcData.output';


tar = tarData.input';
tar(:,17) = tarData.output';


csvwrite('source_park.csv',src);
csvwrite('target_park.csv',tar);
