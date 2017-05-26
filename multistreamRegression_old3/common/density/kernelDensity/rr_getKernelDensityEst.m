function [srcData,tarData] = rr_getKernelDensityEst(srcData,tarData,numPortion,interval)

srcInput = srcData.input;
tarInput = tarData.input;

H_src = getBWMLL(srcInput,numPortion,interval);
H_tar = getBWMLL(tarInput,numPortion,interval);

% get source and target kernel density estimation

srcData.pSrc = getAllKDE(srcInput,H_src,srcInput);
srcData.pTar = getAllKDE(srcInput,H_tar,tarInput);

tarData.pSrc = getAllKDE(tarInput,H_src,srcInput);
tarData.pTar = getAllKDE(tarInput,H_tar,tarInput);

srcData.importance = srcData.pTar./srcData.pSrc;
srcData.weight = srcData.pSrc./srcData.pTar;
tarData.weight = tarData.pSrc./tarData.pTar;

function [pData]=getAllKDE(data,H,input)
numData = size(data,2);
pData = zeros(1,numData);
for i = 1:numData
    x = data(:,i);
    pData(i) = getKDE(x,H,input);
end