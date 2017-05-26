function empiricalMean = rr_getEMean(data)
input = data.input;
output = data.output;
N = size(input,2);
d = size(input,1);
empiricalMean = zeros(d+2,d+2);
for i = 1:N
    x = input(:,i);
    y = output(i); 
    vec = [y;x;1];
    empiricalMean = empiricalMean+vec*vec';
end
empiricalMean = empiricalMean/N;