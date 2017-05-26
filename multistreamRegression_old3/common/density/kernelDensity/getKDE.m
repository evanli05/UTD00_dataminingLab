function KDE_pdf = getKDE(x,H,data)
numData = size(data,2);
d = size(data,1);
KDE_pdf = 0;
for i = 1:numData
    vec = data(:,i);
    KDE_pdf = KDE_pdf+exp(-0.5*(x-vec)'*(H\(x-vec)));
end
KDE_pdf = 1/numData*(2*pi)^(-d/2)*det(H)^(-0.5)*KDE_pdf;