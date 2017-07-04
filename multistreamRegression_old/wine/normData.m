for i = 1:size(wine)
    
%     rowMean = mean(wine(i,:));
%     rowStd = std(wine(i,:));
%     
%     for j = 1:length(wine(i,:))
%         normPark(i,j) = (wine(i,j)-rowMean)/rowStd;
%     end
%     
    rowMax = max(wine(:,i));
    rowMin = min(wine(:,i));
    for j = 1:length(wine(:,i))
        wineQuality(j,i) = (wine(j,i) - rowMin)/(rowMax - rowMin);
    end


end