function [srcData,tarData] = rr_getSrcTarData(D_input,numS,covPortion)
% get dimension of input data
d = size(D_input,1)-1;
numInput = size(D_input,2);

% seperate data randomly and evenly into two parts
% generate a random vector of 1:n - yl
dataIndex = randperm(numInput,numInput);
% break the data evenly - yl
dataIndexS = dataIndex(1:floor(numInput/2));
dataIndexT = dataIndex(floor(numInput/2)+1:end);
D_partS = D_input(:,dataIndexS);
D_partT = D_input(:,dataIndexT);
numInputS = size(D_partS,2);
% numInputT = size(D_partT,2);

% generate source data
% get sample mean and sample variance for the first part data
% mean of each row - yl
sampleMeanS = mean(D_partS(1:d,:),2);
sampleCovS = zeros(d,d);
for i = 1:numInputS
    vec = D_partS(1:d,i);
    sampleCovS = sampleCovS+(vec-sampleMeanS)*(vec-sampleMeanS)';
end
sampleCovS = sampleCovS/(numInputS-1);
sampleCovS = (sampleCovS+sampleCovS')/2;
% generate seek x for source data
x_seekS = (mvnrnd(sampleMeanS',sampleCovS))';
% generate source data
sampleDisS = (mvnpdf(D_partS(1:d,:)',x_seekS',covPortion*sampleCovS))';
% with different weight (distribution generated) - yl
D_src = datasample(D_partS,numS,2,'Replace',false,'Weights',sampleDisS);
% get target data
D_tar = D_partT;

% construct srcData and tarData object
srcData.input = D_src(1:end-1,:);
srcData.output = D_src(end,:);
tarData.input = D_tar(1:end-1,:);
tarData.output = D_tar(end,:);

