%Author : Daniela Cuza
% this script is used to resize the images and labels, and save them in the
% correct format

clear all;



imageDir = 'training\imgs\';
labelDir = 'training\masks\';

imageDir_new = 'dataset_segmentation\imgs\';
labelDir_new = 'dataset_segmentation\masks\';


%PARAMETERS
siz=224; %input size

%%
% Create a |pixelLabelDatastore| holding the ground truth pixel labels for
% the training images.
imds = imageDatastore(imageDir);
pxds = imageDatastore(labelDir);
trainingData = combine(imds,pxds);
data=readall(trainingData);

for i = 1:1010   %Loop for each training image i pass to the function
    data{i,1} =imresize(data{i,1},[siz siz]);
    [A,B,C]=size(data{i,1});
    if C ~= 3
        data{i,1} = cat(3, data{i,1}, data{i,1}, data{i,1});

    end
    lb =data{i,2};
    data{i,2} =imresize(lb(:,:,1),[siz siz]);
end

k=1;
for j = 1:1010
    outI=strcat(imageDir_new,num2str(k),'.png');
    outL=strcat(labelDir_new,num2str(k),'.bmp');
    imwrite(uint8(data{j,1}),outI);
    imwrite(logical(data{j,2}),outL);
    k=k+1;
end






