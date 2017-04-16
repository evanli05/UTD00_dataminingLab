% @author: yifan li

function driftMatrix = drift1(randomArray, nRows)

x1 = randomArray(:,1);
x2 = randomArray(:,2);
x3 = randomArray(:,3);
x4 = randomArray(:,4);
x5 = randomArray(:,5);
delta = randn(nRows,1);
y = 10 * sin(pi .* x1 .* x2) + 20 * (x3 - 0.5).^2 + 10 ...
    * x4 + 5 * x5 + delta;

driftMatrix = [x1, x2, x3, x4, x5, y];