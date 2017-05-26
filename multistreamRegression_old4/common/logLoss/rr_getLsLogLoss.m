function data = rr_getLsLogLoss(data,gammaList,thetaList,estVarList)
[AIWLogLoss,pdfList] = rr_getAIWLogLoss(0,data,gammaList,thetaList,estVarList);
data.lsLogLoss = AIWLogLoss;
data.lsPdf = pdfList;