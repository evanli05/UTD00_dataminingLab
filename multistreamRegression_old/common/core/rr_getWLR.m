function [thetaList,gammaList,estVarList]=rr_getWLR(srcData)

% d = size(D_src,1)-1;
% numS = size(D_src,2);
% gammaList = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
% numGamma = size(gammaList,2);
% % X is a N*(d+1) design matrix with each row (x_i,1)
% X = [D_src(1:d,:)' ones(numS,1)];
% y = D_src(end,:)';
% sampleX = D_src(1:d,:);
% N = size(sampleX,2);
% % get importance
% importanceList = zeros(1,N);
% for i = 1:N
%     x = sampleX(:,i);
%     importanceList(i)=rr_getKDE(x,H_tar,D_tar(1:d,:))/rr_getKDE(x,H_src,D_src(1:d,:));
% end
% % get optimal parameter \theta

d = size(srcData.input,1);
N = size(srcData.input,2);
gammaList = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
numGamma = size(gammaList,2);
% X is a N*(d+1) design matrix with each row (x_i,1)
X = [srcData.input' ones(N,1)];
y = srcData.output';
importanceList = srcData.importance;

thetaList = zeros(d+1,numGamma);
for i = 1:numGamma
    W = diag((importanceList.^(gammaList(i))));
    XTWX = X'*W*X;
    if(rank(XTWX) == min(size(XTWX)))
        %theta = (X'*W*X)\X'*W*y;
        theta = XTWX\X'*W*y;
    else
        theta = pinv(XTWX)*X'*W*y;
    end
    if(isnan(theta))
        thetaList = NaN;
        gammaList = NaN;
        estVarList = NaN;
        return;
    end
    %thetaList(:,i) = (X'*W*X)\X'*W*y
    thetaList(:,i) = theta;
end
% get estimator of variance
estVarList = zeros(1,numGamma);
% estVarList2 = zeros(1,numGamma);
for i = 1:numGamma
    W = diag(importanceList.^(gammaList(i)));
    sumAI = sum(importanceList.^(gammaList(i)));
    estVarList(i) = (y-X*thetaList(:,i))'*W*(y-X*thetaList(:,i))/sumAI;
    % estVarList2(i) = (y-X*thetaList(:,i))'*(y-X*thetaList(:,i))/(N-d-1);
    % estVarList2(i) = (y-X*thetaList(:,i))'*(y-X*thetaList(:,i))/N;
end
% estVarList - estVarList2