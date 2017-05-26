function [meanKDE, varKDE] = rr_getKDEMeanVar(H,data)
d = size(data,1);
N = size(data,2);
meanKDE = mean(data,2);
varKDE = zeros(d,d);
for i = 1:N
    varKDE = varKDE+(data(:,i)-meanKDE)*(data(:,i)-meanKDE)';
end
varKDE = varKDE/N+H;
