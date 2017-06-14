M = csvread('RegSynLocalAbruptDrift_target_streamwhole.csv');
plot(M(:,6));
xticks([9000, 18000, 27000, 36000, 45000, 54000, 63000, 72000, 81000, 90000]);
xticklabels({'9000', '18000', '27000', '36000', '45000', '54000', '63000', '72000', '81000', '90000'})
xlabel('Data Sequence');
ylabel('Value of Output');
title('Local Abrupt Drift Data Set Output')
