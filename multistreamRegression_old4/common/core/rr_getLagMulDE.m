function M = rr_getLagMulDE(srcData,stopThd,rateInitial,decayTune)
format long;
sampleX = srcData.input;
d = size(sampleX,1);
N = size(sampleX,2);
% get the empirical mean via source data
empiricalMean = rr_getEMean(srcData);
% only need [y^2 xy y]
empiricalMean = empiricalMean(:,1);
weight = srcData.weight;
% get low bound of Myy
% lowBMyy = max(-1./(2*varBase*weight));
% initialize lagrangian multipliers
MInitial = zeros(d+2,1);
MInitial(1,1) = 1;
M = MInitial;
% regularization parameter
standardLambda = 1/sqrt(N);
lambda = 0;
% update lagrangian multipliers
rate = rateInitial;
speed = 1;
decay = 1;
% speedThd = 1e-5;
preAvgError = 1e20;
minAvgError = 1e20;
%GM = zeros(d+2,1);
n = 0;
while true
    n=n+1;
    if(n>200000)
        M = NaN;
        break;
    end
    Myy = M(1,1);
    Myx1 = M(2:end,1)';
    appEstMean = zeros(d+2,1);
    for i = 1:N
        x = sampleX(:,i);
        w = weight(i);
        meanY = -Myx1*[x;1]/Myy;
        if(isinf(w))
            varY = 0;
        else
            varY = 1/(2*w*Myy);
        end
        addVarY = zeros(d+2,1);
        addVarY(1,1) = varY;
        appEstMean = appEstMean+[meanY;x;1]*meanY+addVarY;
    end
    appEstMean = appEstMean/N;
    appGradient= appEstMean - empiricalMean - 2*lambda*M;
    avgError = mean(abs(appGradient));
    if(~isreal(avgError) || isnan(avgError))
        M = NaN;
        break;
    end
    if(avgError < stopThd)
        break;
    end
    if(abs(preAvgError - avgError)<1e-15)
        if(minAvgError < 10*stopThd)
            M = bestM;
            break;
        end
        if(lambda == 0)
            lambda = standardLambda;
            rate = rateInitial;
            M = MInitial;
            speed = 1;
            decay = 1;
            % speedThd = 1e-5;
            preAvgError = 1e20;
            minAvgError = 1e20;
            %GM = zeros(d+2,1);
            n = 0;
            continue;
        end
        M = NaN;
        break;
    end
    
    if(avgError < minAvgError)
        minAvgError = avgError;
        %rightGM = GM;
        bestAppGradient = appGradient;
        bestM = M;
    end
    
    
    if(mod(n,100)==1)
        %display(n);
        %rLogLoss = rr_getRobustLogLoss(M,D_tar,D_src,H_src,H_tar,meanBase,varBase);
        %display(['robustLogLoss: ' num2str(rLogLoss,10)]);
        %        display(M');
        %        display(appGradient');
        %GMT = GM';
        %display(GMT);
%         display(['avgError: ' num2str(avgError,10) ' minAvgError: '...
%             num2str(minAvgError,10) ' rate: ' num2str(rate) ...
%             ' decay: ' num2str(decay) ' lambda: ' num2str(lambda)]);
        % if the learning jumps, we recover to the best point and reduce
        % the learning rate
        if(avgError>minAvgError)
            rate = rate*0.9;
            M = bestM;
            appGradient = bestAppGradient;
            avgError = minAvgError;
            %GM = rightGM;
        end
    end
    %GM = GM+appGradient.^2;
    %M = M+rate*decay*speed*GM.^(-1/2).*(-appGradient);
    preM = M;
    M = preM+rate*decay*speed*appGradient;
%     if(M<=0)
%         error('variance is negative!');
%     end
%     comment out by yl

%     diffMyy = M(1,1)-preM(1,1);
%     while(M(1,1)<lowBMyy)
%         M(1,1) = M(1,1)+abs(diffMyy)/2;
%     end
    decay = decayTune/(decayTune+sqrt(n));
    preAvgError = avgError;
end

