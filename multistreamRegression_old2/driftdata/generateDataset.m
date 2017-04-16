% @author: yifan li
% @date: Jan 2, 2017
% @version: 0.1
% @note:
%   1. this program generates the dataset for testing the multistream data
%   with concept drift;
%   2. according to the paper that we are refering to, I present the local 
%   abrupt drift in this program;


function dataConceptDrift = generateDataset()

% i = 1;


for i = 1:3
    dataConceptDriftTemp = zeros(randint(1,1,[500,700]),7);
    nRows = max(size(dataConceptDriftTemp));
    randomArray = rand(nRows,5);
    if i == 1
        dataConceptDriftTemp = drift1(randomArray, nRows);

    elseif i == 2
        dataConceptDriftTemp = drift2(randomArray, nRows);

    else
        dataConceptDriftTemp = drift3(randomArray, nRows);

    end
    
   
    
    % combine matrix together
    if i == 1
        dataConceptDrift{i} = dataConceptDriftTemp;
    else 
        dataConceptDrift{i} = [dataConceptDrift{i-1}; dataConceptDriftTemp];
    end
   
end

dataConceptDrift = dataConceptDrift{3};
    