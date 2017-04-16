% @author: yifan li

function driftMatrix = drift3(randomArray, nRows)

x1 = randomArray(:,1);
x2 = 0.7 + randomArray(:,2) * 0.3;
x3 = 0.7 + randomArray(:,3) * 0.3;
x4 = randomArray(:,4) * 0.3;
x5 = 0.7 + randomArray(:,5) * 0.3;
delta = randn(nRows,1);
y = 10 * cos (x1 .* x2) + 20 * (x3 - 0.5) + exp(x4) + 5 ...
    * x5.^2 + delta;

driftMatrix = [y, x1, x2, x3, x4, x5];