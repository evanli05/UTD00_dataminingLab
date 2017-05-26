function data = rr_getIWLogLoss(data,gammaList,thetaList,estVarList)
[AIWLogLoss,pdfList]=rr_getAIWLogLoss(1,data,gammaList,thetaList,estVarList);
data.lsIWLogLoss = AIWLogLoss;
data.lsIWPdf = pdfList;