%Author : Daniela Cuza
function [data1,data2]=imagesTrasformation_new(data,imageDir,labelDir)
siz=224;
k=7000;
for i = 1:size(data,1)   %Loop for each training image i pass to the function
    data{i,1} =imresize(data{i,1},[siz siz]);
    data{i,2} =imresize(data{i,2},[siz siz]);
    im=data{i,1};   	%consider the image i
    lb=logical(data{i,2}); %consider the label i
    %% Width shift
    interval=224*0.1;  % wide interval: 10%
    a=1+floor(rand*9); % 0-90 % of the image displaced to the right or to the left (9 cases)
    f=rand;
    if f>=0 && f<0.5
        im=imtranslate(im, [interval*a, 0]);  %image displaced to the right
        lb=imtranslate(lb, [interval*a, 0]);
    else
        im=imtranslate(im, [-interval*a, 0]); %image displaced to the left
        lb=imtranslate(lb, [-interval*a, 0]);
    end
    k=saveimLb(imageDir,labelDir,k,im,lb);
    %% Height shift
    im=data{i,1};   	%consider the image i
    lb=logical(data{i,2}); %consider the label i
    a=1+floor(rand*9); % 0-90 % of the image displaced up or down (9 cases)
    f=rand;
    if f>=0 && f<0.5
        im=imtranslate(im, [0, interval*a]);  %image displaced up
        lb=imtranslate(lb, [0, interval*a]);
    else
        im=imtranslate(im, [0, -interval*a]); %image displaced down
        lb=imtranslate(lb, [0, -interval*a]);
    end
    k=saveimLb(imageDir,labelDir,k,im,lb);
    %% Rotation
    im=data{i,1};   	%consider the image i
    lb=logical(data{i,2}); %consider the label i
    ang=45;
    c=1+floor(rand*4); % range 0–180°, with up to 45° intervals
    f=rand;
    if f>=0 && f<0.5
        tform = randomAffine2d('Rotation',[ang*c ang*c]);
    else
        tform = randomAffine2d('Rotation',[-ang*c -ang*c]);
    end
    outputView = affineOutputView(size(im),tform);
    im = imwarp(im,tform,'OutputView',outputView);
    lb = imwarp(lb,tform,'OutputView',outputView);
    k=saveimLb(imageDir,labelDir,k,im,lb);
    
    %% Shear
    im=data{i,1};   	%consider the image i
    lb=logical(data{i,2}); %consider the label i
    c=1+floor(rand);
    d=rand;
    if d>=0 && d<0.25
        tform = randomAffine2d('XShear',[ang*c ang*c]);
    elseif d>=0.25 && d<0.5
        tform = randomAffine2d('XShear',[-ang*c -ang*c]);
    elseif d>=0.5 && d<0.75
        tform = randomAffine2d('YShear',[ang*c ang*c]);
    else
        tform = randomAffine2d('YShear',[-ang*c -ang*c]);     
    end
    outputView = affineOutputView(size(im),tform);
    im = imwarp(im,tform,'OutputView',outputView);
    lb = imwarp(lb,tform,'OutputView',outputView);
    k=saveimLb(imageDir,labelDir,k,im,lb);
    %% flip
    im=data{i,1};   	%consider the image i
    lb=logical(data{i,2}); %consider the label i
    e=rand;
    if d>=0 && d<0.5  % Vertically flip the image
        im=flip(im);
        lb=flip(lb);
    else   % horizontally flip the image
        im=fliplr(im);
        lb=fliplr(lb);
    end
    k=saveimLb(imageDir,labelDir,k,im,lb);
    %% brightness_1
    im=data{i,1};   	%consider the image i
    lb=logical(data{i,2}); %consider the label i
    l=(1+floor(rand))*25;
    ll=rand;
    if ll>=0 && ll<0.5
        im(:,:,1)=im(:,:,1)+l;
        im(:,:,2)=im(:,:,2)+l;
        im(:,:,3)=im(:,:,3)+l;
    else
        im(:,:,1)=im(:,:,1)+(-l);
        im(:,:,2)=im(:,:,2)+(-l);
        im(:,:,3)=im(:,:,3)+(-l);
    end
    k=saveimLb(imageDir,labelDir,k,im,lb);
    %% brightness_2
    im=data{i,1};   	%consider the image i
    lb=logical(data{i,2}); %consider the label i
    u=(1+floor(rand))*25;
    v=(1+floor(rand))*25;
    z=(1+floor(rand))*25;
    uu=rand;
    if uu>=0 && uu<0.5
        im(:,:,1)=im(:,:,1)+u;
        im(:,:,2)=im(:,:,2)+v;
        im(:,:,3)=im(:,:,3)+z;
    else
        im(:,:,1)=im(:,:,1)+(-u);
        im(:,:,2)=im(:,:,2)+(-v);
        im(:,:,3)=im(:,:,3)+(-z);
    end
    k=saveimLb(imageDir,labelDir,k,im,lb);
    %% speckle noise
    im=data{i,1};   	%consider the image i
    lb=logical(data{i,2}); %consider the label i
    im=imnoise(im, 'speckle');
    k=saveimLb(imageDir,labelDir,k,im,lb);

    %% black&white
    im=data{i,1};   	%consider the image i
    lb=logical(data{i,2}); %consider the label i
    im = rgb2gray (im);
    k=saveimLb(imageDir,labelDir,k,im,lb);

    %% equal image
    im=data{i,1};   	%consider the image i
    lb=logical(data{i,2}); %consider the label i
    k=saveimLb(imageDir,labelDir,k,im,lb);
end
data1 = contrast_blur_h_new(data, 1, 1, 1);
data2=shadows_new(data);
end

function k=saveimLb(imageDir,labelDir,k,im,lb)
outI=strcat(imageDir,num2str(k),'.png');
outL=strcat(labelDir,num2str(k),'.bmp');
imwrite(uint8(im),outI);
imwrite(lb,outL);
k=k+1;
end