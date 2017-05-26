function param = trainCovariateShift()


% load data
dataFile = 'dataCDrift';


load(dataFile);
% data = D_input;
data = norDataDrift;

numData = size(data,2);
d = size(data,1)-1;

maxNumTrain = floor(numData/2);
% nTrainList = floor([0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]* maxNumTrain);
nTrainList = floor([0.2]* maxNumTrain);
numTrain = size(nTrainList,2);
% denMethodList = [1 2 3 4];
denMethodList = [1];
numMethod = size(denMethodList,2);

nTrain = nTrainList(1,1);

[src,tar] = rr_getSrcTarData(data,nTrain,0.3);

csvwrite('src.csv',src);
csvwrite('tar.csv',tar);

param = removeShift(src, tar);
end