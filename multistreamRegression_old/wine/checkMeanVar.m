for i=1:12
    colMeanSrc(i) = mean(src(:,i));
    colVarSrc(i) = var(src(:,i));
end

for i=1:12
    colMeanTar(i) = mean(tar(:,i));
    colVarTar(i) = var(tar(:,i));
end

