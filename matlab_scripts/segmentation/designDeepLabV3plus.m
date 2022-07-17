%Author : Daniela Cuza
%% Contruct DeepLab-v3

function net = designDeepLabV3plus(baseModel,inputSize,lastLayer,classWeights,actFun,maxInput)
if nargin==5
	maxInput=1;
end
if nargin==4
    actFun='none';
	maxInput=1;
end
if nargin==3
    classWeights = [1, 1];
    actFun='none';
	maxInput=1;
end


classes = ["one","zero"];
%numClasses = size(classes,2);

net=deeplabv3plusLayersAle([inputSize inputSize], 2, baseModel);
if isequal(lastLayer,'pixel') 
    pxLayer = pixelClassificationLayer('Name','labels','Classes',classes,'ClassWeights',classWeights);
elseif  isequal(lastLayer,'dice')
    pxLayer = dicePixelClassificationLayer('Name','labels');

end
net = replaceLayer(net,"classification",pxLayer);

%if actFun is not none it must change all ReLu layers

switch (actFun)
    case 'none'
        %nothing to do
    case 'random'
        %ResNet50 Active Random
        addpath(genpath('..\VariantiReLu'));
        reluMethods={'tanhReluLayer','meluGaluLayer','flexibleLearnableReluLayer','symmetricMeluLayer','symmetricGaluLayer','splashLayer','widerLearnableReluLayer', 'leakyReluLayer', 'seluLayer' ,'learnableReluLayer','preluLayer' ,'sreluLayer' ,'apluLayer' ,'galuLayer' ,'smallGaluLayer' ,'softLearnable2'  ,'softLearnable'  ,'pdeluLayer'  ,'learnableMishLayer' ,'SRSLayer'   ,'swishLearnable'   ,'swishLayer'};
        [channels, lay] = layersToReplace(net);
        net = ChangeRanAllLayers(channels,lay,reluMethods,net.Layers,net.Connections,1,maxInput);
		
    otherwise
        %ResNet50 Active
        addpath(genpath('..\VariantiReLu'));
        % nameAF='learnableReluLayer'; featureExtractionLayer = 'learnableactivation_49_relu'; skipLayer = 'learnableactivation_10_relu';
        nameAF=actFun;
        %net = resnet50;
        [channels, lay] = layersToReplace(net);
        net = ChangeAllLayers(channels,lay,actFun,net.Layers,net.Connections,1,maxInput);
       
end

end