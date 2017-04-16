logLossMean = infor.logLossMean(:,:,1);

figure;
for i = 1:7
    plot(1:10, logLossMean(i,:));
    xlable()
    hold on;
end