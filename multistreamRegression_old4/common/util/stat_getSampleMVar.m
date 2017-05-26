function [sampleMean, sampleVar] = stat_getSampleMVar(dataVector)
numData = size(dataVector,1);
sampleMean = mean(dataVector);
sampleVar = sum((dataVector - sampleMean).^2)/(numData-1);
