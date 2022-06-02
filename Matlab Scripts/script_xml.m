
%This script is used to save the annotations inside the same folder where
%the image is in the xml format

%In order to execute this code you just need to copy on the folder insider
%the dataset downloaded from http://vision.soic.indiana.edu/projects/egohands/


load('metadata.mat');

%Absolute path of the folder where such script is contained
ABSOLUTE_PATH = "F:/Documenti/FRACAL/Università Francesco/4th Year/Second Semester/Computer Vision/Project/Data/Dataset1/";
IMG_WIDTH = "1280";
IMG_HEIGHT = "720";
OBJECT_NAME = "hand";

%foreach possible folder
for i=1 : length(video)    
    %foreach possible image
    for j=1 : 100
        bounding_boxes = getBoundingBoxes(video(i), j);        
        frame_path = getFramePath(video(i), j);
        parts = split(frame_path,".");        
        path_xml = parts(1,1) + ".xml";
                                
        fileID = fopen(path_xml,"w");        
        fprintf(fileID,"<annotation>\n");
        
        fprintf(fileID,"\t<folder>");        
        parts = split(frame_path,"/");
        fprintf("\t");
        fprintf(fileID,string(parts(length(parts)-1,1)));
        fprintf(fileID,"</folder>\n");
                
        fprintf(fileID,"\t<filename>");    
        fprintf("\t");
        fprintf(fileID,string(parts(length(parts),1)));
        fprintf(fileID,"</filename>\n");
        
        fprintf(fileID,"\t<path>");
        fprintf("\t");
        fprintf(fileID,strcat(ABSOLUTE_PATH,frame_path));
        fprintf(fileID,"</path>\n");
                
        fprintf(fileID,"\t<source>\n");
        fprintf(fileID,"\t\t<database>");
        fprintf(fileID,"Unknown");
        fprintf(fileID,"</database>\n");
        fprintf(fileID,"\t</source>\n");
        
        fprintf(fileID,"\t<size>\n");        
        fprintf(fileID,"\t\t<width>");  
        fprintf(fileID,IMG_WIDTH);
        fprintf(fileID,"</width>\n");        
        fprintf(fileID,"\t\t<heigth>");
        fprintf(fileID,IMG_HEIGHT);
        fprintf(fileID,"</heigth>\n");
        fprintf(fileID,"\t\t<depth>");
        fprintf(fileID,"3");
        fprintf(fileID,"</depth>\n");        
        fprintf(fileID,"\t</size>\n");
        
        fprintf(fileID,"\t<segmented>");
        fprintf(fileID,"0");
        fprintf(fileID,"</segmented>\n");
        
        
        
        
        
        dimensions = size(bounding_boxes);        
        for row = 1 : dimensions(1,1)            
            if all(bounding_boxes(row,:))                                  
                xmin = string(bounding_boxes(row,1));
                ymin = string(bounding_boxes(row,2));
                xmax = string(bounding_boxes(row,1) + bounding_boxes(row,3));
                ymax = string(bounding_boxes(row,2) + bounding_boxes(row,4));
                    
                fprintf(fileID,"\t<object>\n");
                fprintf(fileID,"\t\t<name>");
                fprintf(fileID,OBJECT_NAME);
                fprintf(fileID,"</name>\n");
                fprintf(fileID,"\t\t<pose>");
                fprintf(fileID,"Unspecified");
                fprintf(fileID,"</pose>\n");
                fprintf(fileID,"\t\t<trucated>");
                fprintf(fileID,"0");
                fprintf(fileID,"</trucated>\n");
                fprintf(fileID,"\t\t<difficult>");
                fprintf(fileID,"0");
                fprintf(fileID,"</difficult>\n");
                 
                fprintf(fileID,"\t\t<bndbox>\n");
                fprintf(fileID,"\t\t\t<xmin>");
                fprintf(fileID,xmin);
                fprintf(fileID,"</xmin>\n");
                fprintf(fileID,"\t\t\t<ymin>");
                fprintf(fileID,ymin);
                fprintf(fileID,"</ymin>\n");
                fprintf(fileID,"\t\t\t<xmax>");
                fprintf(fileID,xmax);
                fprintf(fileID,"</xmax>\n");
                fprintf(fileID,"\t\t\t<ymax>");
                fprintf(fileID,ymax);
                fprintf(fileID,"</ymax>\n");
                fprintf(fileID,"\t\t</bndbox>\n");                 
                fprintf(fileID,"\t</object>\n");
                
            end                       
        end
        fprintf(fileID,"</annotation");
        fclose(fileID);        
    end        
end




