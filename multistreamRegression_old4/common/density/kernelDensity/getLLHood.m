function lLHood = getLLHood(H,evaluateData,trainData)
numData = size(evaluateData,2);
lLHood =0;
for i = 1:numData
    vector = evaluateData(:,i);
    logPdf = log(getKDE(vector,H,trainData));
    lLHood = lLHood + logPdf;
end
end