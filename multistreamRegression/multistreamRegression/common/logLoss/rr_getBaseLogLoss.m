function data = rr_getBaseLogLoss(data,meanBase,varBase)
%load('initialSet');
input = data.input;
output = data.output;
N = size(input,2);
basePdf = zeros(1,N);
baseLogLoss = 0;
for i = 1:N
    y = output(i);
    logPdf = log(1/(sqrt(varBase)*sqrt(2*pi)))-(y-meanBase)^2/(2*varBase);
    basePdf(1,i) = exp(logPdf);
    baseLogLoss = baseLogLoss+logPdf;
end
baseLogLoss = -baseLogLoss/N;
data.baseLogLoss = baseLogLoss;
data.basePdf = basePdf;
