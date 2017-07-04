for i=1:17
    colMeanSrc(i) = mean(src(:,i));
    colVarSrc(i) = var(src(:,i));
end

for i=1:17
    colMeanTar(i) = mean(tar(:,i));
    colVarTar(i) = var(tar(:,i));
end

