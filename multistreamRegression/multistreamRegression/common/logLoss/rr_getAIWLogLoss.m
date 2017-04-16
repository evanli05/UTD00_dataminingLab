function [AIWLogLoss,AIWPdf] = rr_getAIWLogLoss(gamma,data,gammaList,thetaList,estVarList)
input = data.input;
output = data.output;
N = size(input,2);
AIWPdf = zeros(1,N);

index = find(gammaList==gamma,1);
theta = thetaList(:,index);
varY = estVarList(index);
AIWLogLoss = 0;
for i=1:N
    x = [input(:,i);1];
    y = output(i);
    meanY = x'*theta;
    pdf = mvnpdf(y,meanY,varY);
    if(pdf == 0)
        display('pdf is almost zero');
    end
    logPdf = log(1/(sqrt(varY)*sqrt(2*pi)))-(y-meanY)^2/(2*varY);
    AIWPdf(1,i) = exp(logPdf);
    AIWLogLoss = AIWLogLoss+logPdf;
end
AIWLogLoss = -AIWLogLoss/N;

