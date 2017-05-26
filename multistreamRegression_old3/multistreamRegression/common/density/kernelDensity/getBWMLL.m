function H = getBWMLL(data,numPortion,interval)
numData = size(data,2);
if(numPortion > numData)
    error('not enough data');
end
portion = round(numData/numPortion);
sampleCov = cov(data');
alpha = 1;
bestAlpha = 1;
while(alpha>0)
    H = alpha*sampleCov;
    logLHood = 0;
    for j = 1:numPortion
        indexStart = (j-1)*portion+1;
        if(j<numPortion)
            indexEnd = j*portion;
        else
            indexEnd = numData;
        end
        evaluateData = data(:,indexStart:indexEnd); 
        trainData = data;
        trainData(:,indexStart:indexEnd)=[];
        logLHood = logLHood + getLLHood(H,evaluateData,trainData);
    end
    logLHood = logLHood/numPortion;
    if(alpha == 1)
        bestLogLHood = logLHood;
    end
    if(logLHood > bestLogLHood)
        bestLogLHood = logLHood;
        bestAlpha = alpha;
    end
    alpha = alpha - interval;
end
H = bestAlpha*sampleCov;
end