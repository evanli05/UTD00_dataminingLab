
driftcases = -1;
driftcases = 1;
driftcases = 2;
driftcases = -1;


switch driftcases
    case -1
        src(max(size(src))+1:max(size(src))+244,1:11) = src(1:244, 1:11)
        src(max(size(src))+1:max(size(src))+244,12) = src(1:244, 12) - 1
    case 1
        src(max(size(src))+1:max(size(src))+244,1:11) = src(1:244, 1:11)
        src(max(size(src))+1:max(size(src))+244,12) = src(1:244, 12) + 1
    case 2
        src(max(size(src))+1:max(size(src))+244,1:11) = src(1:244, 1:11)
        src(max(size(src))+1:max(size(src))+244,12) = src(1:244, 12) + 2
end