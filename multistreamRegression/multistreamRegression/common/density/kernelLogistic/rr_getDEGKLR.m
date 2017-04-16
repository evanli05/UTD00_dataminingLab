function [srcData,tarData] = rr_getDEGKLR(srcData,tarData,lambda,rbfScale)
% design training data for logistic regression
% y=-1 is source and y=1 is target
options.Display = 'false';
srcInput = srcData.input;
nSrc = size(srcInput,2);
% d = size(srcInput,1);
XSrc = [ones(nSrc,1) srcInput'];
ySrc = -ones(nSrc,1);
tarInput = tarData.input;
nTar = size(tarInput,2);
XTar = [ones(nTar,1) tarInput'];
yTar = ones(nTar,1);

X = [XSrc;XTar];
y = [ySrc;yTar];

% Squared exponential radial basis function kernel expansion
nInstances = nSrc+nTar;
% lambda = bestR/sqrt(nInstances);
% rbfScale = 1;
Krbf = kernelRBF(X,X,rbfScale);
funObj = @(u)LogisticLoss(u,Krbf,y);
fprintf('Training kernel(rbf) logistic regression model...\n');
uRBF = minFunc(@penalizedKernelL2,zeros(nInstances,1),options,Krbf,funObj,lambda);

KrbfSrc = kernelRBF(XSrc,X,rbfScale);
KrbfTar = kernelRBF(XTar,X,rbfScale);
srcData.importance = (nSrc./nTar).*exp(KrbfSrc*uRBF);
srcData.importance = srcData.importance';
srcData.weight = 1./srcData.importance;
tarData.weight = 1./((nSrc./nTar).*exp(KrbfTar*uRBF));
tarData.weight = tarData.weight';
% pSrc = 1./(1+exp(KrbfSrc*uRBF)));
% pTar = 1-1./(1+exp(KrbfTar*uRBF));
