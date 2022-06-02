
%This script is used to save the annotations inside the same folder where
%the image is in the format supported by yolo network i.e. :
%centerx,centery, widthx, widthy all relative to the dimension of the whole
%image i.e. whole image dimensions = 1

%In order to execute this code you just need to copy on the folder insider
%the dataset downloaded from http://vision.soic.indiana.edu/projects/egohands/


load('metadata.mat');

%foreach possible folder

fileID = fopen("hands.csv","w");        


for i=1 : length(video)    
    %foreach possible image
    for j=1 : 100
        bounding_boxes = getBoundingBoxes(video(i), j);        
        frame_path = getFramePath(video(i), j);
        parts = split(frame_path,"/");     
        
        file_name_extension = parts(length(parts),1);                        
         fprintf(fileID,strcat(file_name_extension,","));
        dimensions = size(bounding_boxes);        
         %x and y mark the top left corner of the box. w and h    
        row_to_print = "";
        for row = 1 : dimensions(1,1)                              
            
               
            minx = bounding_boxes(row,1);
            miny = bounding_boxes(row,2);
            maxx = minx + bounding_boxes(row,3);
            maxy = miny + bounding_boxes(row,4);
                   
            row_to_print= strcat(row_to_print,num2str(minx),",");
            row_to_print= strcat(row_to_print,num2str(miny), ",");
            row_to_print= strcat(row_to_print,num2str(maxx), ",");
              
            if row == dimensions(1,1)
                row_to_print= strcat(row_to_print,num2str(maxy),"\n");                
            else
                row_to_print= strcat(row_to_print,num2str(maxy),",");
            end                
            
                                   
        end      
        fprintf(fileID,row_to_print);
    end        
end
fclose(fileID);  




