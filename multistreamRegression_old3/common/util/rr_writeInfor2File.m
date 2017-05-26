function rr_writeInfor2File(infor,filePath)
fid = fopen(filePath,'w');
string = ['name of data:' infor.dataName];
fprintf(fid,'%s\n',string);
string = ['number of data:' num2str(infor.numData)];
fprintf(fid,'%s\n',string);
string = ['number of attributes:' num2str(infor.numAttr)];
fprintf(fid,'%s\n',string);
string = ['various training number:' num2str(infor.nTrainList)];
fprintf(fid,'%s\n',string);

fprintf(fid,'\n%s\n','liner logistic regression');
fclose(fid);
dlmwrite(filePath,infor.logLossMean(:,:,1),'delimiter','\t','-append');
fid = fopen(filePath,'a');
fprintf(fid,'\n%s\n','L2 liner logistic regression via 10 cross validation');
fclose(fid);
dlmwrite(filePath,infor.logLossMean(:,:,2),'delimiter','\t','-append');
fid = fopen(filePath,'a');
fprintf(fid,'\n%s\n','gaussian kernel logistic regression');
fclose(fid);
dlmwrite(filePath,infor.logLossMean(:,:,3),'delimiter','\t','-append');
fid = fopen(filePath,'a');
fprintf(fid,'\n%s\n','L2 gaussian kernel logistic regression via 10 cross validation');
fclose(fid);
dlmwrite(filePath,infor.logLossMean(:,:,4),'delimiter','\t','-append');
