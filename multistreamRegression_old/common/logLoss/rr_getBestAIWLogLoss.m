function data = rr_getBestAIWLogLoss(data,gammaList,thetaList,estVarList)
[AIWLogLoss,pdfList] = rr_getAIWLogLoss(0.2,data,gammaList,thetaList,estVarList);
BAIWLogLoss = AIWLogLoss;
BPdfList = pdfList;
for gamma = [0.3 0.4 0.5 0.6 0.7 0.8 0.9]
    [AIWLogLoss,pdfList] = rr_getAIWLogLoss(gamma,data,gammaList,thetaList,estVarList);
    if(BAIWLogLoss>AIWLogLoss)
        BAIWLogLoss = AIWLogLoss;
        BPdfList = pdfList;
    end
end
data.lsBAIWLogLoss = BAIWLogLoss;
data.lsBAIWPdf = BPdfList;
end