function data = rr_getRobustKLDLogLoss(M,data,meanBase,varBase)
input = data.input;
output = data.output;
weight = data.weight;
N = size(input,2);
% d = size(D_tar,1)-1;
% numT = size(D_tar,2);
Myy = M(1,1);
%Myx1 = M(1,2:end);
Myx1 = M(2:end,1)';
robustKLDLogLoss = 0;
robustKLDPdf = zeros(1,N);
%baseLogLoss = 0;
for i=1:N
    x = input(:,i);
    %x = D_tar(1:d,i);
    y = output(i);
    %y = D_tar(end,i);
    w = weight(i);
    if(isinf(w))
        robustKLDLogLoss = NaN;
	break;
    end 
    meanY = 1/(2*w*Myy+(1/varBase))*(-2*w*Myx1*[x;1]+(1/varBase)*meanBase);
    varY = 1/(2*w*Myy+(1/varBase));
    if(~(varY>0))
      robustKLDLogLoss = NaN;
      break;
    end 
    pdf = mvnpdf(y',meanY',varY);
    if(pdf == 0)
        display('pdf is almost zero!');
        %error('pdf is zero!');
        %continue;
    end
    logPdf = log(1/(sqrt(varY)*sqrt(2*pi)))-(y-meanY)^2/(2*varY);
    robustKLDPdf(1,i) = exp(logPdf);
    if(~isreal(logPdf))
        robustKLDLogLoss = NaN;
        break;
    end
    robustKLDLogLoss = robustKLDLogLoss+logPdf;
end
robustKLDLogLoss = -robustKLDLogLoss/N;
data.robustKLDLogLoss = robustKLDLogLoss;
data.robustKLDPdf = robustKLDPdf;
