function [meanBase,varBase] = rr_getDisBase(y_min,y_max)
% estimate mean and variance of base distribution
meanBase = (y_min+y_max)/2;
varBase = ((y_max-meanBase)/2)^2;