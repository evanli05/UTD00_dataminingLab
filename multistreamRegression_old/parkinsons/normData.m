for i = 1:17
    
%     rowMean = mean(parkinsons(i,:));
%     rowStd = std(parkinsons(i,:));
%     
%     for j = 1:length(parkinsons(i,:))
%         normPark(i,j) = (parkinsons(i,j)-rowMean)/rowStd;
%     end
%     
    rowMax = max(parkinsons(i,:));
    rowMin = min(parkinsons(i,:));
    for j = 1:length(parkinsons(i,:))
        park(i,j) = (parkinsons(i,j) - rowMin)/(rowMax - rowMin);
    end


end