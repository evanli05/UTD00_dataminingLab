clear all;
close all;
clc

driftDataset = importdata('driftData_input.mat');
for i = 1:6
    valMean = mean(driftDataset(i,:));
    valStd = std(driftDataset(i,:));
    driftDataset(i,:) = (driftDataset(i,:) - valMean) / valStd;
end