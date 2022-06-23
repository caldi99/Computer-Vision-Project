%This script is used to save the annotations inside the same folder where
%the image is in the format : LABEL X Y W H

%load stuff
addpath('../egohands_dataset/');
load('../egohands_dataset/metadata.mat');

for i=1 : length(video)    
    for j=1 : 100
        %get bounding boxes
        bounding_boxes = getBoundingBoxes(video(i), j);        
        
        %create name of the file.txt
        frame_path = getFramePath(video(i), j);        
        parts = split(frame_path,".");        
        path_txt = "../egohands_dataset/" + parts(1,1) + ".txt";                        
        
        %create file
        fileID = fopen(path_txt,"w");        
        
        %parse bounding boxes
        dimensions = size(bounding_boxes);         
        for row = 1 : dimensions(1,1)            
            if all(bounding_boxes(row,:))  
                row_to_print = "0 ";                
                x = bounding_boxes(row,1);
                y = bounding_boxes(row,2);
                w = bounding_boxes(row,3);
                h = bounding_boxes(row,4);                    
                row_to_print= strcat(row_to_print,num2str(x), " ");
                row_to_print= strcat(row_to_print,num2str(y), " ");
                row_to_print= strcat(row_to_print,num2str(w), " ");                
                if row == dimensions(1,1)
                    row_to_print= strcat(row_to_print,num2str(h));
                else
                    row_to_print= strcat(row_to_print,num2str(h),"\n");
                end                
                fprintf(fileID,row_to_print);
            end                       
        end
        
        %close file
        fclose(fileID);        
    end        
end