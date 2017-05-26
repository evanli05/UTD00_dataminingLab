function conf = rr_getConf(matrix,z)
numCol = size(matrix,2);
diagCov = diag(cov(matrix'));
conf = z*sqrt(diagCov/numCol);