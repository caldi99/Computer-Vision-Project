"""
    This file was entirly created by Francesco Caldivezzi
"""

# ------------------------------------------------------------------
# ZIPS PATHS AND DESTINATIONS
# ------------------------------------------------------------------

#Zip folders and files
ZIP_EGOHANDS = "../egohands_data.zip"
TARGZ_HANDSOVERFACE = "../hand_over_face_corrected.tar.gz"
ZIP_TESTSET = "../drive-download-20220530T161656Z-001.zip"

#Destinations zips
DESTINATION_ZIP_EGOHANDS = "../egohands_dataset/"
DESTINATION_TARGZ_HANDSOVERFACES = "../handsoverfaces/"
DESTINATION_ZIP_TESTSET = "../testset/"

# ------------------------------------------------------------------
# DESTINATIONS DIRECTORIES PATHS FOR THE TWO DATASET
# ------------------------------------------------------------------

#dataset/ Directory
DATASET_DIR = "../dataset/"

#dataset/dataset_handsoverfaces/ Directory 
DATASET_HANDSOVERFACES_DIR = "../dataset/dataset_handsoverfaces/"
#dataset/dataset_handsoverfaces/images/ Directory
DATASET_HANDSOVERFACES_IMAGES_DIR = "../dataset/dataset_handsoverfaces/images/"
#dataset/dataset_handsoverfaces/annotations/ Directory
DATASET_HANDSOVERFACES_ANNOTATIONS_DIR = "../dataset/dataset_handsoverfaces/annotations/"
#dataset/dataset_handsoverfaces/images/hand/ Directory
DATASET_HANDSOVERFACES_IMAGES_HAND_DIR = "../dataset/dataset_handsoverfaces/images/hand/"
#dataset/dataset_handsoverfaces/images/no_hand/ Directory
DATASET_HANDSOVERFACES_IMAGES_NOHAND_DIR = "../dataset/dataset_handsoverfaces/images/no_hand/"
#dataset/dataset_handsoverfaces/images/raw/ Directory
DATASET_HANDSOVERFACES_IMAGES_RAW_DIR = "../dataset/dataset_handsoverfaces/images/raw/"

#dataset/dataset_egohands/ Directory
DATASET_EGOHANDS_DIR = "../dataset/dataset_egohands/"
#dataset/dataset_egohands/images/ Directory
DATASET_EGOHANDS_IMAGES_DIR = "../dataset/dataset_egohands/images/"
#dataset/dataset_egohands/annotations/ Directory
DATASET_EGOHANDS_ANNOTATIONS_DIR = "../dataset/dataset_egohands/annotations/"
#dataset/dataset_egohands/images/hand/ Directory
DATASET_EGOHANDS_IMAGES_HAND_DIR = "../dataset/dataset_egohands/images/hand/"
#dataset/dataset_egohands/images/no_hand/ Directory
DATASET_EGOHANDS_IMAGES_NOHAND_DIR = "../dataset/dataset_egohands/images/no_hand/"
#dataset/dataset_handsoverfaces/images/raw/ Directory
DATASET_EGOHANDS_IMAGES_RAW_DIR = "../dataset/dataset_egohands/images/raw/"

# ------------------------------------------------------------------
# DESTINATIONS DIRECTORIES PATHS FOR DATASET FOR TRAINING
# ------------------------------------------------------------------

#dataset/dataset/ Directory
DATASET_DATASET_DIR = "../dataset/dataset/"
#dataset/dataset/train/ Directory
DATASET_DATASET_TRAIN_DIR = "../dataset/dataset/train/"
#dataset/dataset/train/hand/ Directory
DATASET_DATASET_TRAIN_HAND_DIR = "../dataset/dataset/train/hand/"
#dataset/dataset/train/no_hand/ Directory
DATASET_DATASET_TRAIN_NOHAND_DIR = "../dataset/dataset/train/no_hand/"
#dataset/dataset/validation/ Directory
DATASET_DATASET_VALIDATION_DIR = "../dataset/dataset/validation/"
#dataset/dataset/validation/hand/ Directory
DATASET_DATASET_VALIDATION_HAND_DIR = "../dataset/dataset/validation/hand/"
#dataset/dataset/validation/no_hand/ Directory
DATASET_DATASET_VALIDATION_NOHAND_DIR = "../dataset/dataset/validation/no_hand/"
#dataset/dataset/test/ Directory
DATASET_DATASET_TEST_DIR = "../dataset/dataset/test/"
#dataset/dataset/test/hand/ Directory
DATASET_DATASET_TEST_HAND_DIR = "../dataset/dataset/test/hand/"
#dataset/dataset/test/no_hand/ Directory
DATASET_DATASET_TEST_NOHAND_DIR = "../dataset/dataset/test/no_hand/"

# ------------------------------------------------------------------
# OTHER STUFF
# ------------------------------------------------------------------

WIDTH_IMG_EGOHANDS_DATASET = 1280
HEIGTH_IMG_EGOHANDS_DATASET = 720

#_LABELLED_SAMPLES_EGOHANDS Directory
LABELLED_SAMPLES_DIR = "../egohands_dataset/_LABELLED_SAMPLES/"
#Folders to ignore egohands dataset, notice that, the first 9 folders are ignored because contains images of the test set
FOLDERS_TO_IGNORE_EGOHANDS = ["CARDS_LIVINGROOM_B_T",
                            "CARDS_OFFICE_H_T",
                            "CARDS_OFFICE_B_S"
                            "CHESS_COURTYARD_B_T",
                            "CHESS_OFFICE_B_S",                            
                            "PUZZLE_COURTYARD_B_S",                            
                            "PUZZLE_OFFICE_B_H",
                            "JENGA_LIVINGROOM_H_B",
                            "JENGA_COURTYARD_B_H",                            
                            "CARDS_COURTYARD_B_T"] #poor images inside here

#handsoverfaces/hand_over_face/annotations/ Directtory
HANDSOVERFACES_HANDOVERFACE_ANNOTATIONS = "../handsoverfaces/hand_over_face/annotations/"

#images_original_size Directory
IMAGES_ORIGINAL_SIZE_DIR = "../handsoverfaces/hand_over_face/images_original_size/" 
#Files to ignore hand over face dataset first 10 files are ignored because they are inside the test set
FILES_TO_IGNORE = ["1.jpg",
                    "2.jpg",
                    "3.jpg",
                    "178.jpg",
                    "34.jpg",
                    "38.jpg",
                    "53.jpg",
                    "77.jpg",
                    "146.jpg",
                    "245.jpg",
                    "216.jpg", #problematic files
                    "221.jpg"]



#Xmls handsoverfaces
DIR_HANDSOVERFACES_ANNOTATIONS = "../handsoverfaces/hand_over_face/annotations/"

#Dir Original Size Images Handsoverfaces
DIR_ORIGINALSIZE_IMAGES_HANDSOVERFACES = "../handsoverfaces/hand_over_face/images_original_size/"

# ------------------------------------------------------------------
# CNN CONFIGS
# ------------------------------------------------------------------

#Input CNN Dimensions
INPUT_DIMS = (224,224)
INPUT_TENSOR_DIMS = (224,224,3)

#80% Validation (25% of 90%) + Training (75% of 90%) 10% Test
TRAIN_PERCENTAGE = 0.75
VALIDATION_PERCENTAGE = 0.25
TEST_PERCENTAGE = 0.1

#Learing rate
INIT_LR = 1e-4

#Parameters Datagerator
DATAGENERATOR_RESCALE = 1./255
DATAGENERATOR_ROTATIONRANGE = 20
DATAGENERATOR_ZOOMRANGE = 0.15
DATAGENERATOR_WIDTHSHIFTRANGE = 0.2
DATAGENERATOR_HEIGTHSHIFTRANGE = 0.2
DATAGENERATOR_SHEARRANGE = 0.15

#EPOCHS
EPOCHS = 5

#BATCH SIZE
BS = 32

#MODEL PATH
MODEL_PATH = "model.h5"
MODEL_PATH_PB = "model.pb"

#TRAIN VALIDATION TEST PATHS
TRAIN_PATH = "dataset/train/"
VALIDATION_PATH = "dataset/validation/"
TEST_PATH = "dataset/test/" 

#PATH PLOT
PATH_PLOT = "plot.png"

#PATH FOLDER MODEL OPENCV/TENSORFLOW
MODEL_PATH_PB_DIR = "model/"