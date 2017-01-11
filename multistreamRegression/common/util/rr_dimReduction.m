function D_X = rr_dimReduction(dataSet,explainedVar)
[~, score,~,~, explained] = pca(dataSet');
% select k components explain 90% of variance
explainedVar = explainedVar*100;
cuExpVar = 0;
k = 0;
while cuExpVar < explainedVar
    k = k+1;
    cuExpVar = cuExpVar + explained(k);
end
D_X = score(:,1:k);
D_X = D_X';
