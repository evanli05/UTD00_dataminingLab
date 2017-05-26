% @author: yifan li

function driftMatrix = drift2(randomArray, nRows)

% x1 ~ uniform [0,1], x2, x3, x5 ~ uniform [0, 0.3], x4 ~
% uniform [0.7,1], delta ~ normal (0,1)
x1 = randomArray(:,1);
x2 = randomArray(:,2) * 0.3;
x3 = randomArray(:,3) * 0.3;
x4 = 0.7 + randomArray(:,4) * 0.3;
x5 = randomArray(:,5) * 0.3;
delta = randn(nRows,1);
y = 10 .* x1 .* x2 + 20 * (x3 - 0.5) + 10 * x4 + 5 * x5 ...
    + delta;

driftMatrix = [x1, x2, x3, x4, x5, y];