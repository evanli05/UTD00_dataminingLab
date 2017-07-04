clear;
%add path
path = '../common/';
path = genpath(path);
addpath(path);
dataName = 'wine';
% load data
dataFile = [dataName '_input'];


load(dataFile);
data = wine;
% number of instances
numData = size(data,2);
% number of independent variables
d = size(data,1)-1; 

maxNumTrain = floor(numData/2);
% nTrainList = floor([0.2 0.4 0.6 0.8 1]*maxNumTrain);
nTrainList = floor([0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]*maxNumTrain);
numTrain = size(nTrainList,2);
% number of method, here also choose the method
denMethodList = [1 2 3 4];
numMethod = size(denMethodList,2);

nR = 10;
% used to be 100;

infor.dataName = dataName;
infor.numData = numData;
infor.numAttr = d;
infor.nTrainList = nTrainList;
infor.logLossMean = zeros(2,5,4);
infor.conf = zeros(2,5,4);
% infor.logLossMean = zeros(7,10,4);
% infor.conf = zeros(7,10,4);

% for each method the theory want to use, run the loop once and save the
% results
for k = 1:numMethod
    denMethod = denMethodList(1,k);
%     for j = 1:numTrain % comment
        nTrain = nTrainList(1,1); % used to be j
%         initialization of the fail count
        iFail = 0;
        rDEFail = 0;
        rKLDFail = 0;
        logLossMatrix = zeros(2,1);
        i=0;
        while i <= nR-1
            i = i+1;
            %% ------------- get source and target dataSet --------------
            [srcData,tarData] = rr_getSrcTarData(data,nTrain,0.3);
            nSrc = size(srcData.input,2);
            nTar = size(tarData.input,2);
            nInstances = nSrc+nTar;
            standardLambda = 1/sqrt(nInstances);
            
            %% ------------------ density estimation --------------------
            %%    kernel density estimation
            
%             % originally commented out - yl
%                 crossV = 10;
%                 interval = 0.02;
%                 [srcData,tarData] = rr_getKernelDensityEst(srcData,tarData,crossV,interval);
%             
            %%    linear logistic regression
            if(denMethod == 1)
                %   get denstiy ratio via specified L2 regulator
                lambda = 0;
                [srcData,tarData] = rr_getDELLR(srcData,tarData,lambda);
                
                % %     get density ratio via L2 regulator by cross validation method
            elseif(denMethod == 2)
                crossV = 10;
                lambdaList = [0 0.1 1 3 10 30 100 300 1000].*standardLambda;
                [srcData,tarData,bestR] = rr_getDELLRCV(srcData,tarData,crossV,lambdaList);
                
                %%    Gaussian kernel logistic regression
            elseif(denMethod == 3)
                %     get denstiy ratio via specified L2 regulator
                lambda = standardLambda;
                rbfScale = 1;
                [srcData,tarData] = rr_getDEGKLR(srcData,tarData,lambda,rbfScale);
                
                %     get density ratio via L2 regulator by cross validation method
            elseif(denMethod == 4)
                crossV = 10;
                lambdaList = [1 3 10 30 100 300 1000].*standardLambda;
                [srcData,tarData,bestR] = rr_getDEGKLRCV(srcData,tarData,crossV,lambdaList);
            end
            
            %%  --------------- methods -----------------
            %%  baseline method
            y_min = min(srcData.output);
            y_max = max(srcData.output);
            [meanBase,varBase] = rr_getDisBase(y_min,y_max);
            tarData = rr_getBaseLogLoss(tarData,meanBase,varBase);
            baseLogLoss = tarData.baseLogLoss;
            
            %%  linear method
            % get theta estimated variance for different gamma
            [thetaList,gammaList,estVarList] = rr_getWLR(srcData);
            if(isnan(thetaList))
                iFail = iFail+1;
                i = i-1;
                continue;
            end
            
            tarData = rr_getLsLogLoss(tarData,gammaList,thetaList,estVarList);
            lsLogLoss = tarData.lsLogLoss;
            
            tarData = rr_getBestAIWLogLoss(tarData,gammaList,thetaList,estVarList);
            lsBAIWLogLoss = tarData.lsBAIWLogLoss;
            
            tarData = rr_getIWLogLoss(tarData,gammaList,thetaList,estVarList);
            lsIWLogLoss = tarData.lsIWLogLoss;
            
            tarData = rr_getBLogLoss(srcData,tarData,1*eye(d+1),estVarList(:,1));
            lsBLogLoss = tarData.lsBLogLoss;
            
            %%  robust differential entropy method
            % get the lagrangian multiplier matrix
            % the last three parameters stopThd, rateInitial, decayTune
            MDE = rr_getLagMulDE(srcData,1e-2,1e-2,300);
            if(isnan(MDE))
                rDEFail = rDEFail + 1;
                i = i - 1;
                continue;
            end
            tarData = rr_getRobustDELogLoss(MDE,tarData);
            robustDELogLoss = tarData.robustDELogLoss;
            
            %%  robust Kullback-Leibler divergence method
            % get the lagrangian multiplier matrix
            % the last three parameters stopThd, rateInitial, decayTune
            MKLD = rr_getLagMulKLD(srcData,meanBase,varBase,1e-2,1e-2,300);
            if(isnan(MKLD))
                rKLDFail = rKLDFail + 1;
                i = i - 1;
                continue;
            end
            tarData = rr_getRobustKLDLogLoss(MKLD,tarData,meanBase,varBase);
            robustKLDLogLoss = tarData.robustKLDLogLoss;
            
            %% save result
            
            logLossMatrix(:,i)=[robustKLDLogLoss;baseLogLoss];
%             logLossMatrix(:,i)=[robustDELogLoss;robustKLDLogLoss;baseLogLoss;lsLogLoss;lsBAIWLogLoss;lsIWLogLoss;lsBLogLoss];
            %   display(tarData);
            display([i j k]);
            display(mean(logLossMatrix,2));
        end
        
        logLossMean = mean(logLossMatrix,2);
        conf = rr_getConf(logLossMatrix,1.96);
        infor.logLossMean(:,j,k) = logLossMean;
        infor.conf(:,j,k) = conf;
        save([dataName 'Infor'],'infor');
        
%     end % comment
    filePath = [dataName 'Infor' '.txt'];
    rr_writeInfor2File(infor,filePath);
end
