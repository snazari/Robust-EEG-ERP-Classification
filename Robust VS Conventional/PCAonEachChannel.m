%% output=PCAonEachChannel(data)
% this function applies pca on each channels of data for all trials matrix
% and remove dimension with eigen value less than minimumRelativeEigenvalue
%       Input:
%           data- (number of time samples X number of channels X number of trials)
%
%       Output:
%           output- backprojected data with removed low variance components
%%
function output=PCAonEachChannel(data)

minimumRelativeEigenvalue=1e-05;
[d1,d2,d3]=size(data);

for i=1:d2
    data1=squeeze(data(:,i,:)) ;
    data1=bsxfun(@minus,data1,mean(data1,2));
    [U,E]=eig(data1*data1');
    eigenvalues=diag(E);
    selectedEigenvalues=((eigenvalues(end)*minimumRelativeEigenvalue)<=eigenvalues);
    projectionMatrix=zeros(d1);
    projectionMatrix(selectedEigenvalues,:)=U(:,selectedEigenvalues).';
    data2(:,i,:)=projectionMatrix*data1;
end

output=data2;
end

