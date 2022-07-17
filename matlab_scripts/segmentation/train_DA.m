%Author : Daniela Cuza
clear all;

outputDir='output\';

imageDir = 'dataset_segmentation\imgs\';
labelDir= 'dataset_segmentation\masks\';

imageDir_new = 'AUG_mia_new\imgs\';
labelDir_new = 'AUG_mia_new\masks\';

imageValDir = 'test_new\rgb\';
labelValDir = 'test_new\mask\';

reluMethod='none'; %usa ReLU
modelName='res18';
% 
gpuDevice(1)


%PARAMETRI
siz=224; %input size
weights=1; % weight classes
lr=0.01 ; %learning rate
maxEpoch =20; %number of epochs
opt='sgdm'; %'adam'; %optimizer
aug=1; %data augmentation
loss='dice';%pixel'; %'dice ' ; % loss function
rete='resnet18';
MiniBatchSize=25; %50


if weights==1
     classWeights =   [3.2464; 0.5910]; 
else
     classWeights = [1; 1];    
end   
lgraph = designDeepLabV3plus(rete,siz,loss,classWeights,reluMethod,255);
%% 
% Create a |pixelLabelDatastore| holding the ground truth pixel labels for 
% the training images.
imds = imageDatastore(imageDir);
pxds = imageDatastore(labelDir);
trainingData = combine(imds,pxds);
data=readall(trainingData);
[data1,data2]=imagesTrasformation_new(data,imageDir_new,labelDir_new);
saveImLb(data1,23000,imageDir_new,labelDir_new);
saveImLb(data2,25000,imageDir_new,labelDir_new);


imds_new= imageDatastore(imageDir);
pxds_new = pixelLabelDatastore(labelDir,["one","zero"],[1 0]);

ds = pixelLabelImageDatastore(imds_new,pxds_new, 'OutputSize',[siz siz] );


%%options
options = trainingOptions(opt, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',5,...
    'LearnRateDropFactor',0.2,...
    'Momentum',0.9, ...
    'InitialLearnRate',lr, ...
    'L2Regularization',0.005, ...
    'MaxEpochs',maxEpoch, ...  
    'MiniBatchSize',MiniBatchSize, ...
    'Shuffle','every-epoch', ...
    'VerboseFrequency',5,...
    'Plots','training-progress'); 
    %'ValidationData',dsVal,... 
    %'ValidationPatience', 5,...
	
%% 
% Train the network.
net = trainNetwork(ds,lgraph,options);
modelDir=strcat(outputDir,'models\');
if ~exist(modelDir, 'dir')
    mkdir(modelDir);
end
save(strcat(modelDir,modelName),'net');

%%
%Test model
maskDir=strcat(outputDir,'outmask\',modelName,'\');
%calculate marks (in not exists)
if ~exist(maskDir, 'dir')
    mkdir(maskDir);
    saveMask(net, imageValDir,maskDir,siz);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function saveMask(net, imageDir,outDir,siz)
   
imds = imageDatastore(imageDir);
numImages = numel(imds.Files);
    
                
for i=1:numImages
    img = readimage(imds, i);
    
    [bMask,mask]=getMask(net, img,siz);
    [~,name,~] = fileparts(imds.Files{i});
    %save PNG in the output dir
    outfile=strcat(outDir,name,'.png');
    imwrite(mask,outfile);
	
    %save BMP in the output dir
    outfile=strcat(outDir,name,'.bmp');
    imwrite(bMask,outfile);      
end
    
end

%to save binary mask
function [bMask,mask]=getMask(net, img,siz)
    origSize=size(img);
    % seg ment
    
    img=imresize(img,[siz siz]);
    
    [C,~,allScores]  = semanticseg(img, net);
    mask=1-allScores(:,:,1);
    mask=imresize(mask, [origSize(1) origSize(2)]);
    %save BMP in the output dir
    bMask=(C=='one');
    bMask=imresize(bMask, [origSize(1) origSize(2)]);
    
end

function saveImLb(trainingData,n,imageValDir,labelValDir)
    for i=1:size(trainingData,1)
        outI=strcat(imageValDir,num2str(i+n),'.png');
        outL=strcat(labelValDir,num2str(i+n),'.bmp');
        imwrite(uint8(trainingData{i,1}),outI);
        imwrite(logical(im2bw(trainingData{i,2})),outL);
    end
end