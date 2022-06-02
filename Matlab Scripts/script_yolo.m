
%This script is used to save the annotations inside the same folder where
%the image is in the format supported by yolo network i.e. :
%centerx,centery, widthx, widthy all relative to the dimension of the whole
%image i.e. whole image dimensions = 1

%In order to execute this code you just need to copy on the folder insider
%the dataset downloaded from http://vision.soic.indiana.edu/projects/egohands/


load('metadata.mat');

%foreach possible folder
for i=1 : length(video)    
    %foreach possible image
    for j=1 : 100
        bounding_boxes = getBoundingBoxes(video(i), j);        
        frame_path = getFramePath(video(i), j);
        parts = split(frame_path,".");        
        path_txt = parts(1,1) + ".txt";                        
        fileID = fopen(path_txt,"w");        
        dimensions = size(bounding_boxes);        
         %x and y mark the top left corner of the box. w and h                
        for row = 1 : dimensions(1,1)            
            if all(bounding_boxes(row,:))  
                row_to_print = "0 ";
                
                x = bounding_boxes(row,1);
                y = bounding_boxes(row,2);
                w = bounding_boxes(row,3);
                h = bounding_boxes(row,4);
                    
                centerx = round((x + (w/2)) / 1280, 6);
                centery = round((y + (h/2)) / 720, 6);
                xcorrect = round(x/1280,6);
                ycorrect = round(x/720,6);
                    
                row_to_print= strcat(row_to_print,num2str(centerx), " ");
                row_to_print= strcat(row_to_print,num2str(centery), " ");
                row_to_print= strcat(row_to_print,num2str(xcorrect), " ");
                
                if row == dimensions(1,1)
                    row_to_print= strcat(row_to_print,num2str(ycorrect));
                else
                    row_to_print= strcat(row_to_print,num2str(ycorrect),"\n");
                end                
                fprintf(fileID,row_to_print);
            end                       
        end
        fclose(fileID);        
    end        
end




