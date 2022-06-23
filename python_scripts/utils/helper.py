"""
    This file was entirly created by Francesco Caldivezzi
"""
import cv2
from bs4 import BeautifulSoup
from config import configs
import os
import numpy as np
import shutil
import glob
import random
import zipfile
import tarfile

class Helper:
    """
        Helper class used to carry out some of the most pre processing steps before the training of the Resnet50V2 network
    """

    # -----------------------------------------------------------------------------
    # EXTRACTION DATASET METHODS
    # -----------------------------------------------------------------------------
    
    def extract_file_zip(self,path_zip_file,output_dir):
        """
            Extract a .zip file into a specified directory
            Parameters
            ---------
            path_zip_file : str
                            The path of the file zip
            output_dir : str
                            The path of the output directory
        """
        with zipfile.ZipFile(path_zip_file, 'r') as zip:
            zip.extractall(output_dir)
    
    def extract_file_targz(self,path_targz_file,output_dir):
        """
            Extract a .tar.gz file into a specified directory
            Parameters
            ---------
            path_zip_file : str
                            The path of the file .tar.gz
            output_dir : str
                            The path of the output directory
        """
        file = tarfile.open(path_targz_file)
        file.extractall(output_dir)
        file.close()
    
    # -----------------------------------------------------------------------------
    # SIMPLE HELPER FUNCTIONS
    # -----------------------------------------------------------------------------
    
    def copy_images(self,list_paths,to_dir):
        """
            Copy images given their path to to_dir
            Parameters
            ---------
            list_paths : list of str
                        The list of paths of the images
            to_dir : str
                        The path of the directory where to copy the images
        """
        i=0
        for path in list_paths:
            new_path ="{}{}.jpg".format(to_dir,i)            
            shutil.copy(path,new_path)
            i = i+1

    def remove_file(self,path_file):
        """
            Remove a file given its path
            Parameters
            ---------
            path_file : of str
                        Path of the file to remove
        """        
        os.remove(path_file)

    def is_image(self,name_image):
        """
            Understand if name_image parameter is an .jpg file
            Parameters
            ---------
            name_image : of str
                        Name of a file
            Returns
            -------
            bool
                True if name_image is indeed a .jpg file False otherwise
        """  
        if(len(name_image.split('.jpg')) > 1):
            return True
        else:
            return False
    
    def is_txt(self,name_txt):
        """
            Understand if name_txt parameter is an .txt file
            Parameters
            ---------
            name_txt : of str
                        Name of a file
            Returns
            -------
            bool
                True if name_txt is indeed a .txt file False otherwise
        """  
        if(len(name_txt.split('.txt')) > 1):
            return True
        else:
            return False

    def check_coordinates(self,x1_gen, y1_gen, x2_gen, y2_gen, filename, heigth, width,is_egohands):
        """
            Check if the provided coordinates overlap the ground truth of the image identified by filename
            Parameters
            ---------
            x1_gen : int
                        Top left x coordinate to check
            x2_gen : int
                        Top left y coordinate to check
            y1_gen : int
                        Bottom rigth x coordinate to check
            y2_gen : int
                        Bottom rigth y coordinate to check
            filename : str
                        Name of the file for which checking the provided coordinates
            heigth : int
                        Heigth of the image identified by filename
            width : int
                        Width of the image identified by filename
            is_egohands : bool
                        True if we are dealing with egohand dataset False if we are dealing with handoverface dataset
            Returns
            -------
            bool
                True if the provided coordinates are ok False otherwise
        """  
        if(is_egohands):
            file = open(configs.DATASET_EGOHANDS_ANNOTATIONS_DIR + filename + ".txt","r")
        else:
            file = open(configs.DATASET_HANDSOVERFACES_ANNOTATIONS_DIR + filename + ".txt","r")
        
        matrix = np.zeros((heigth,width))        
        for line in file:
            if(line != "\n"):
                parts = line.split(" ")
                x1 = int(parts[1])
                y1 = int(parts[2])
                x2 = x1 + int(parts[3])
                y2 = y1 + int(parts[4])
                matrix[y1 : y2 +1, x1: x2 +1] = 1
        if(matrix[y1_gen:y2_gen+1,x1_gen:x2_gen+1].sum() != 0):
            return False
        else:
            return True

    # -----------------------------------------------------------------------------
    # ANNOTATIONS CONVERTER FUNCTION
    # -----------------------------------------------------------------------------
    
    def save_annotations_txt_format_handsoverfaces(self):
        """
            Save the annotations in txt format for hand over faces dataset            
        """
        #Parse xml file
        list_xmls = os.listdir(configs.DIR_HANDSOVERFACES_ANNOTATIONS)
        for xml in list_xmls:            
            counturs = []
            contents = open(configs.DIR_HANDSOVERFACES_ANNOTATIONS + xml).read()
            soup = BeautifulSoup(contents, "html.parser")            
            for countur in soup.find_all("polygon"):
                counturs.append([])
                for point in countur.find_all("pt"):
                    counturs[len(counturs)-1].append([int(point.find("x").string),int(point.find("y").string)])
            
            #Add counturs to array of counturs
            points = []
            for c in counturs:
                points.append((np.array(c , dtype=np.int32 )))

            #Start Generation of hands from images
            name_file = xml.split(".xml")[0]
            image_color = cv2.imread(configs.DIR_ORIGINALSIZE_IMAGES_HANDSOVERFACES + name_file + ".jpg")
            image_gray = cv2.imread(configs.DIR_ORIGINALSIZE_IMAGES_HANDSOVERFACES + name_file + ".jpg",0)
            
            #Create and open new file
            file = open("{}{}.txt".format(configs.DIR_HANDSOVERFACES_ANNOTATIONS,name_file),"w")

            for countur in points:
                try:
                    #Create mask and identify hand in the image
                    countur_format_opencv = [np.array(countur , dtype=np.int32)]
                    mask = np.zeros_like(image_gray)
                    cv2.drawContours(mask, countur_format_opencv , -1, 255, -1)
                    out = np.zeros_like(image_gray)
                    out[mask == 255] = image_gray[mask == 255]

                    #Get Coordinates bounding box
                    (y, x) = np.where(mask == 255)
                    (topy, topx) = (np.min(y), np.min(x))
                    (bottomy, bottomx) = (np.max(y), np.max(x))                
                    out = image_color[topy:bottomy+1, topx:bottomx+1]

                    #Save if only valid coordinates
                    if(topx != bottomx and topy != bottomy):
                        #Save File
                        h = bottomy - topy 
                        w = bottomx - topx                         
                        file.write("0 {} {} {} {}\n".format(topx,topy,w,h))                                            
                except cv2.error as e:
                    pass               
            #Close file
            file.close()

    # -----------------------------------------------------------------------------
    # MOVE DATASET FUNCTIONS
    # -----------------------------------------------------------------------------
    
    def move_egohands_dataset(self,from_dir,to_dir,folders_to_ignore):
        """
            Move the egohand dataset from "from_dir" to "to_dir" ignoring the images of the dataset that are inside "folders_to_ignore"
             Parameters
            ---------
            from_dir : str
                        Path of the directory _LABELLED_SAMPLES of the egohand dataset
            to_dir : str
                        Path of the destination directory
            folders_to_ignore : list of str
                                List of folders inside _LABELLED_SAMPLES directory that we don't want to consider
        """
        listdir = sorted(os.listdir(from_dir))        
        i = 0        
        for dir in listdir:
            if(dir not in folders_to_ignore):
                elements = sorted(os.listdir(os.path.join(from_dir,dir)))
                for element in elements:
                    if(self.is_image(element)):
                        old_path = os.path.join(from_dir,dir,element)
                        new_path = os.path.join(to_dir,"image" + str(i) + ".jpg" )
                        i = i + 1                    
                        shutil.copy(old_path,new_path)
    
    def move_handoverface_dataset(self,from_dir,to_dir,files_to_ignore):
        """
            Move the handoverface dataset from "from_dir" to "to_dir" ignoring the images with names inside "files_to_ignore"
             Parameters
            ---------
            from_dir : str
                        Path of the directory images_original_size
            to_dir : str
                        Path of the destination directory
            files_to_ignore : list of str
                                List of names of images to ignore
        """
        listdir = sorted(os.listdir(from_dir)) 
        i = 0
        for image in listdir:
            if(self.is_image(image) and image not in files_to_ignore):
                old = os.path.join(from_dir,image)
                new = os.path.join(to_dir,"image" + str(i)+".jpg")   
                i = i+1             
                shutil.copy(old,new)
    
    def move_egohands_annoatations(self,from_dir,to_dir,folders_to_ignore):
        """
            Move the egohand annotations from "from_dir" to "to_dir" ignoring the annotations of the dataset that are inside "folders_to_ignore"
            Parameters
            ---------
            from_dir : str
                        Path of the directory _LABELLED_SAMPLES of the egohand dataset
            to_dir : str
                        Path of the destination directory
            folders_to_ignore : list of str
                                List of folders inside _LABELLED_SAMPLES directory that we don't want to consider
        """
        listdir = sorted(os.listdir(from_dir))        
        i = 0        
        for dir in listdir:
            if(dir not in folders_to_ignore):
                elements = sorted(os.listdir(os.path.join(from_dir,dir)))
                for element in elements:
                    if(self.is_txt(element)):
                        old_path = os.path.join(from_dir,dir,element)
                        new_path = os.path.join(to_dir, "image" + str(i) + ".txt" )
                        i = i + 1                    
                        shutil.copy(old_path,new_path)
    
    def move_handoverface_annoatations(self,from_dir,to_dir,files_to_ignore):
        """
            Move the handoverface annotations from "from_dir" to "to_dir" ignoring the annotations with names inside "files_to_ignore"
             Parameters
            ---------
            from_dir : str
                        Path of the directory annotations
            to_dir : str
                        Path of the destination directory
            files_to_ignore : list of str
                                List of names of annotations to ignore
        """
        listdir = sorted(os.listdir(from_dir)) 
        i = 0
        for image in listdir:
            if(self.is_txt(image) and (image.split('.txt')[0] + ".jpg") not in files_to_ignore):
                old = os.path.join(from_dir,image)
                new = os.path.join(to_dir,"image" + str(i)+".txt")   
                i = i+1             
                shutil.copy(old,new)

    # -----------------------------------------------------------------------------
    # CREATE POSITIVE AND NEGATIVE IMAGES FUNCTIONS
    # -----------------------------------------------------------------------------

    def create_positive_egohandsdataset(self):
        """
            Create positive Samples for the egohand dataset
        """
        list_annotations = sorted(os.listdir(configs.DATASET_EGOHANDS_ANNOTATIONS_DIR))        
        for txt in list_annotations:
            i = 0
            #Retrieve the corresponding image
            image_name = txt.split(".txt")[0] + ".jpg"
            image = cv2.imread(configs.DATASET_EGOHANDS_IMAGES_RAW_DIR + image_name)
            #Open file
            file = open(configs.DATASET_EGOHANDS_ANNOTATIONS_DIR + txt,"r")
            for line in file:
                if line != "\n":
                    parts = line.split(" ")
                    x = int(parts[1])
                    y = int(parts[2])
                    w = int(parts[3])
                    h = int(parts[4])
                    crop = image[y : y+h, x : x+w]
                    crop = cv2.resize(crop,configs.INPUT_DIMS,cv2.INTER_CUBIC)
                    cv2.imwrite("{}{}_{}.jpg".format(configs.DATASET_EGOHANDS_IMAGES_HAND_DIR,txt.split(".txt")[0],i) ,crop)
                    i = i+1
            file.close()

    def create_positive_handsoverfacesdatset(self):
        """
            Create positive Samples for the handoverface dataset
        """
        list_annotations = sorted(os.listdir(configs.DATASET_HANDSOVERFACES_ANNOTATIONS_DIR))        
        for txt in list_annotations:
            i = 0
            #Retrieve the corresponding image
            image_name = txt.split(".txt")[0] + ".jpg"
            image = cv2.imread(configs.DATASET_HANDSOVERFACES_IMAGES_RAW_DIR + image_name)
            #Open file
            file = open(configs.DATASET_HANDSOVERFACES_ANNOTATIONS_DIR + txt,"r")
            for line in file:
                if line != "\n":
                    parts = line.split(" ")
                    x = int(parts[1])
                    y = int(parts[2])
                    w = int(parts[3])
                    h = int(parts[4])                    
                    crop = image[y : y+h, x : x+w]
                    crop = cv2.resize(crop,configs.INPUT_DIMS,cv2.INTER_CUBIC)                    
                    cv2.imwrite("{}{}_{}.jpg".format(configs.DATASET_HANDSOVERFACES_IMAGES_HAND_DIR,txt.split(".txt")[0],i) ,crop)
                    i = i+1
            file.close()

    def create_negative_egohandsdatset(self):
        """
            Create negative Samples for the egohand dataset
        """
        list_images = os.listdir(configs.DATASET_EGOHANDS_IMAGES_RAW_DIR)
        for name_image in list_images:            
            image = cv2.imread(configs.DATASET_EGOHANDS_IMAGES_RAW_DIR + name_image)
            i=0
            for row in range(0,720,248):
                for col in range(0,1024,160):                   
                    if self.check_coordinates(col, row, col + 224, row +224, name_image.split('.jpg')[0],configs.HEIGTH_IMG_EGOHANDS_DATASET,configs.WIDTH_IMG_EGOHANDS_DATASET,True):
                        negative = image[row : row + 224, col : col + 224]
                        cv2.imwrite("{}{}_{}.jpg".format(configs.DATASET_EGOHANDS_IMAGES_NOHAND_DIR,name_image.split('.jpg')[0],i) ,negative)
                        i=i+1

    def create_negative_handsoverfaces(self):
        """
            Create negative Samples for the handoverface dataset
        """
        list_images = os.listdir(configs.DATASET_HANDSOVERFACES_IMAGES_RAW_DIR)
        for name_image in list_images:            
            image = cv2.imread(configs.DATASET_HANDSOVERFACES_IMAGES_RAW_DIR + name_image)            
            heigth_box = int((224 * image.shape[0])/(720))
            width_box = int((224 * image.shape[1])/(1280))            
            i=0
            for row in range(0,image.shape[0],heigth_box):
                for col in range(0,image.shape[1],width_box):                   
                    if self.check_coordinates(col, row, col + width_box, row +heigth_box, name_image.split('.jpg')[0],image.shape[0],image.shape[1],False):
                        negative = image[row : row + heigth_box, col : col + width_box]
                        crop = cv2.resize(negative, configs.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite("{}{}_{}.jpg".format(configs.DATASET_HANDSOVERFACES_IMAGES_NOHAND_DIR,name_image.split('.jpg')[0],i), crop)
                        i=i+1
    
    # -----------------------------------------------------------------------------
    # CREATION OF DATASET
    # -----------------------------------------------------------------------------

    def creating_dataset_cnn(self):
        """
            This function create the dataset for the training of the Resnet50V2 CNN
        """
        #Get list of paths images
        hand_list_path = glob.glob(configs.DATASET_HANDSOVERFACES_IMAGES_HAND_DIR + "*.*")
        hand_list_path =  hand_list_path + glob.glob(configs.DATASET_EGOHANDS_IMAGES_HAND_DIR + "*.*")
        nohand_list_path = glob.glob(configs.DATASET_HANDSOVERFACES_IMAGES_NOHAND_DIR + "*.*")
        nohand_list_path =  nohand_list_path + glob.glob(configs.DATASET_EGOHANDS_IMAGES_NOHAND_DIR + "*.*")
        
        #Shuffle list of paths
        random.shuffle(hand_list_path)
        random.shuffle(nohand_list_path)

        #Sizes Computation
        test_hand_size = int(len(hand_list_path) * configs.TEST_PERCENTAGE)
        test_nohand_size = int(len(nohand_list_path) * configs.TEST_PERCENTAGE)
        remaining_length_hand = len(hand_list_path) - test_hand_size
        remaining_length_nohand = len(nohand_list_path) - test_nohand_size
        train_hand_size = int(remaining_length_hand * configs.TRAIN_PERCENTAGE)
        train_nohand_size = int(remaining_length_nohand * configs.TRAIN_PERCENTAGE)
        
        #Path computation
        test_hand_list_path = hand_list_path[0:test_hand_size]
        test_nohand_list_path = nohand_list_path[0:test_nohand_size]
        remaining_hand_list_path = hand_list_path[test_hand_size:]
        remaining_nohand_list_path = nohand_list_path[test_nohand_size:]
        train_hand_list_path = remaining_hand_list_path[0:train_hand_size]
        train_nohand_list_path = remaining_nohand_list_path[0:train_nohand_size]
        val_hand_list_path = remaining_hand_list_path[train_hand_size:]        
        val_nohand_list_path = remaining_nohand_list_path[train_nohand_size:]

        #Copy Images
        self.copy_images(train_hand_list_path,configs.DATASET_DATASET_TRAIN_HAND_DIR)        
        self.copy_images(train_nohand_list_path,configs.DATASET_DATASET_TRAIN_NOHAND_DIR)
        self.copy_images(val_hand_list_path,configs.DATASET_DATASET_VALIDATION_HAND_DIR)        
        self.copy_images(val_nohand_list_path,configs.DATASET_DATASET_VALIDATION_NOHAND_DIR)
        self.copy_images(test_hand_list_path,configs.DATASET_DATASET_TEST_HAND_DIR)
        self.copy_images(test_nohand_list_path,configs.DATASET_DATASET_TEST_NOHAND_DIR)

    def remove_unnecessary_things(self):
        """
            This function remove egohands/ and handsoverfaces/ folders
        """
        shutil.rmtree(configs.DESTINATION_TARGZ_HANDSOVERFACES, False, None)
        shutil.rmtree(configs.DESTINATION_ZIP_EGOHANDS, False, None)
