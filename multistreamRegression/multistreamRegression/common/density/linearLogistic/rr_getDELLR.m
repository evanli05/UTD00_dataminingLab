function [srcData,tarData] = rr_getDELLR(srcData,tarData,lambda)
% design training data for logistic regression
% y=-1 is source and y=1 is target
options.Display = 'false';
srcInput = srcData.input;
nSrc = size(srcInput,2);
d = size(srcInput,1);
XSrc = [ones(nSrc,1) srcInput'];
ySrc = -ones(nSrc,1);
tarInput = tarData.input;
nTar = size(tarInput,2);
XTar = [ones(nTar,1) tarInput'];
yTar = ones(nTar,1);

X = [XSrc;XTar];
y = [ySrc;yTar];
nVars = d;

% linear logistic regression
funObj = @(w)LogisticLoss(w,X,y);
lambda = lambda*ones(nVars+1,1);
lambda(1) = 0; % Don't penalize bias
%fprintf('Training L2-regularized logistic regression model...\n');
wL2 = minFunc(@penalizedL2,zeros(nVars+1,1),options,funObj,lambda);
pSrc = 1./(1+exp(XSrc*wL2));
pTar = exp(XTar*wL2)./(1+exp(XTar*wL2));
srcData.importance = (nSrc./nTar).*exp(XSrc*wL2);
srcData.importance = srcData.importance';
srcData.weight = 1./srcData.importance;
tarData.weight = 1./((nSrc./nTar).*exp(XTar*wL2));
tarData.weight = tarData.weight';
% trainErr_L2 = sum(y ~= sign(X*wL2))/length(y);
