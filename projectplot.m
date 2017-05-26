% Big Data Plot
clear all;
close all;

removeCSCDLogloss = [2.12, 2.07, 2.09, 2.35, 2.20, 2.25, 2.17, 2.25, 2.14, 2.13];
normalLogloss = [3.12, 3.20, 3.08, 3.52, 3.63, 3.55, 3.50, 3.34, 3.36, 3.32];

xAxis = linspace(1,10,10);

figure(1);
plot(xAxis, removeCSCDLogloss, xAxis, normalLogloss);

% hold on;
% plot(xAxis, normalLogloss);