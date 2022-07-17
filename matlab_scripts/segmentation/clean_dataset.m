%Author : Daniela Cuza

% this script is useful to clean the dataset, in particular it finds the
% masks that are black
pathL = 'dataset_segmentation\masks\';
pathIm = 'dataset_segmentation\imgs\';

for i = 1:1010

    label = strcat(pathL, num2str(i), '.bmp');
    image = strcat(pathIm, num2str(i), '.png');
    lb = imread(label);
    if sum(sum(lb,1),2) <= 8
        delete(label)
        delete(image)
    end
end
