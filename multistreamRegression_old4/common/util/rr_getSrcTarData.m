function [srcData,tarData] = rr_getSrcTarData(D_input,numS,covPortion)
% get dimension of input data
d = size(D_input,1)-1;
% numInput = size(D_input,2)/3;

% seperate data randomly and evenly into two parts
for j = 1:3
dataIndexS = linspace((j-1)*500+1,(j-1)*500+500,500);
D_partS = D_input(:,dataIndexS);
numInputS = size(D_partS,2);
% numInputT = size(D_partT,2);

% generate source data
% get sample mean and sample variance for the first part data
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
D_partS(7,:) = sampleDisS;
D_partSS = zeros(7,1);
D_partTT = zeros(7,1);
for k = 1:500
    if D_partS(7,k)<0.001
        D_partSS = [D_partSS, D_partS(:,k)];
    else
        D_partTT = [D_partTT, D_partS(:,k)];
    end
end
    
D_src = datasample(D_partSS(1:6,:),numS,2,'Replace',true,'Weights',D_partSS(7,:));
% get target data
D_tar = datasample(D_partTT(1:6,:),numS*9,2,'Replace',true,'Weights',D_partTT(7,:));

if j == 1
        dataDriftSrc{j} = D_src;
        dataDriftTar{j} = D_tar;
    else 
        dataDriftSrc{j} = [dataDriftSrc{j-1} D_src];
        dataDriftTar{j} = [dataDriftTar{j-1} D_tar];
    end

% construct srcData and tarData object

% dataDriftSrc = D_src;
% dataDriftTar = D_tar;

end


srcData = dataDriftSrc{3};
% srcData = dataDriftSrc{1};
tarData = dataDriftTar{3};
% tarData = dataDriftTar{1};
