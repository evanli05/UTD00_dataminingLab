function data = rr_getRobustDELogLoss(M,data)
input = data.input;
output = data.output;
weight = data.weight;
N = size(input,2);
Myy = M(1,1);
Myx1 = M(2:end,1)';
robustDELogLoss = 0;
robustDEPdf = zeros(1,N);
for i=1:N
    x = input(:,i);
    %x = D_tar(1:d,i);
    y = output(i);
    %y = D_tar(end,i);
    w = weight(i);
    if(isinf(w))
        robustDELogLoss = NaN;
	break;
    end
    meanY = -Myx1*[x;1]/Myy;
    varY = 1/(2*w*Myy);
    % meanY = 1/(2*w*Myy+(1/varBase))*(-2*w*Myx1*[x;1]+(1/varBase)*meanBase);
    % varY = 1/(2*w*Myy+(1/varBase));
    if(~(varY>0))
      robustDELogLoss = NaN;
      break;
    end 
    pdf = mvnpdf(y',meanY',varY);
    if(pdf == 0)
        display('pdf is almost zero!');
        %error('pdf is zero!');
        %continue;
    end
    logPdf = log(1/(sqrt(varY)*sqrt(2*pi)))-(y-meanY)^2/(2*varY);
    robustDEPdf(1,i) = exp(logPdf);
    if(~isreal(logPdf))
        robustDELogLoss = NaN;
        break;
    end
    robustDELogLoss = robustDELogLoss+logPdf;
end
robustDELogLoss = -robustDELogLoss/N;
data.robustDELogLoss = robustDELogLoss;
data.robustDEPdf = robustDEPdf;