function tarData = rr_getBLogLoss(srcData,tarData,priorVar,noiseVar)
numS = size(srcData.input,2);
X = [srcData.input' ones(numS,1)];
ySrcList = srcData.output';
A = (1/noiseVar)*(X'*X)+inv(priorVar);
numT = size(tarData.input,2);
BPdfList = zeros(1,numT);
BLogLoss = 0;
for i = 1:numT
    x_tar = [tarData.input(:,i);1];
    y_tar = tarData.output(i);
    meanY = (1/noiseVar)*x_tar'*(A\(X'*ySrcList));
    postVar = x_tar'*(A\x_tar);
    varY = postVar+noiseVar;
    pdf = mvnpdf(y_tar,meanY,varY);
    if(pdf == 0)
        display('error');
    end
    logPdf = log(1/(sqrt(varY)*sqrt(2*pi)))-(y_tar-meanY)^2/(2*varY);
    BPdfList(1,i) = exp(logPdf);
    BLogLoss = BLogLoss+logPdf;
end
BLogLoss = -BLogLoss/numT;
tarData.lsBLogLoss = BLogLoss;
tarData.lsBPdf = BPdfList;