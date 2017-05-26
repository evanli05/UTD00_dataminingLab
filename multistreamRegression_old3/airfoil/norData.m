load('datadrift.mat')
norDataDrift = zeros(6,1500);
for i = 1:6
    rowMean = mean(dataConceptDrift(i,:));
    rowVariance = var(dataConceptDrift(i,:));
    norDataDrift(i,:) = (dataConceptDrift(i,:)-rowMean)/sqrt(rowVariance);
end

save('dataCDrift', 'norDataDrift')