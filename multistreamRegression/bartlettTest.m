clear all;
close all;
clc;

rng default  % for reproducibility
mu = [0 0];
sigma = [1 0.99; 0.99 1];
X = mvnrnd(mu,sigma,20);  % columns 1 and 2
X(:,3:4) = mvnrnd(mu,sigma,20);  % columns 3 and 4
X(:,5:6) = mvnrnd(mu,sigma,20);  % columns 5 and 6

[ndim, prob] = barttest(X,0.05)