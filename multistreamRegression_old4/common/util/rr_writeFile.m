function rr_writeFile(fileID,nTrain,nR,logLossMean,conf)
fprintf(fileID,'%8s %4s\n','nTrain','nR');
fprintf(fileID,'%8d %4d\n',nTrain,nR);
fprintf(fileID,'%10s\n','logLossMean');
fprintf(fileID,'%10.10f\n',logLossMean);
fprintf(fileID,'%10s\n','95% confidence');
fprintf(fileID,'%10.10f\n',conf);