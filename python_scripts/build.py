"""
    This file was entirely created by Francesco Caldivezzi
"""

from distutils.command.config import config
from utils.helper import Helper
from config import configs
import os

helper = Helper()

# -------------------------------------------------------------------------------
# EXTRACTION OF FILES
# -------------------------------------------------------------------------------

print("EXTRACTING {} TO {}".format(configs.ZIP_EGOHANDS,configs.DESTINATION_ZIP_EGOHANDS))
helper.extract_file_zip(configs.ZIP_EGOHANDS,configs.DESTINATION_ZIP_EGOHANDS)

print("EXTRACTING {} TO {}".format(configs.ZIP_TESTSET,configs.DESTINATION_ZIP_TESTSET))
helper.extract_file_zip(configs.ZIP_TESTSET,configs.DESTINATION_ZIP_TESTSET)

print("EXTRACTING {} TO {}".format(configs.TARGZ_HANDSOVERFACE,configs.DESTINATION_TARGZ_HANDSOVERFACES))
helper.extract_file_targz(configs.TARGZ_HANDSOVERFACE,configs.DESTINATION_TARGZ_HANDSOVERFACES)

print("REMOVING {}".format(configs.ZIP_EGOHANDS))
helper.remove_file(configs.ZIP_EGOHANDS)

print("REMOVING {}".format(configs.TARGZ_HANDSOVERFACE))
helper.remove_file(configs.TARGZ_HANDSOVERFACE)

print("REMOVING {}".format(configs.ZIP_TESTSET))
helper.remove_file(configs.ZIP_TESTSET)

# -------------------------------------------------------------------------------
# EXECUTION OF MATLAB FILE
# -------------------------------------------------------------------------------

word = input("PLEASE TYPE Y IF YOU HAVE EXECUTED THE matlab_scripts/create_annotations_egohands_dataset.m FILE OTHERWISE, EXECUTE IT AND THEN TYPE Y")
while(word != "Y" and word != "y"):
    word = input("PLEASE TYPE Y IF YOU HAVE EXECUTED THE matlab_scripts/create_annotations_egohands_dataset.m FILE OTHERWISE, EXECUTE IT AND THEN TYPE Y")

# -------------------------------------------------------------------------------
# CREATION OF DIRECTORIES
# -------------------------------------------------------------------------------

print("CREATION OF NECESSARY DIRECTORIES")

#Create directory dataset/
if not os.path.exists(configs.DATASET_DIR):
    os.mkdir(configs.DATASET_DIR)

#Create directory dataset/dataset_handsoverfaces/
if not os.path.exists(configs.DATASET_HANDSOVERFACES_DIR):
    os.mkdir(configs.DATASET_HANDSOVERFACES_DIR)
#Create directory dataset/dataset_handsoverfaces/images/
if not os.path.exists(configs.DATASET_HANDSOVERFACES_IMAGES_DIR):
    os.mkdir(configs.DATASET_HANDSOVERFACES_IMAGES_DIR)
#Create directory dataset/dataset_handsoverfaces/annotations/
if not os.path.exists(configs.DATASET_HANDSOVERFACES_ANNOTATIONS_DIR):
    os.mkdir(configs.DATASET_HANDSOVERFACES_ANNOTATIONS_DIR)
#Create directory dataset/dataset_handsoverfaces/images/hand/
if not os.path.exists(configs.DATASET_HANDSOVERFACES_IMAGES_HAND_DIR):
    os.mkdir(configs.DATASET_HANDSOVERFACES_IMAGES_HAND_DIR)
#Create directory dataset/dataset_handsoverfaces/images/no_hand/
if not os.path.exists(configs.DATASET_HANDSOVERFACES_IMAGES_NOHAND_DIR):
    os.mkdir(configs.DATASET_HANDSOVERFACES_IMAGES_NOHAND_DIR)
#Create directory dataset/dataset_handsoverfaces/images/raw/
if not os.path.exists(configs.DATASET_HANDSOVERFACES_IMAGES_RAW_DIR):
    os.mkdir(configs.DATASET_HANDSOVERFACES_IMAGES_RAW_DIR)

#Create directory dataset/dataset_egohands/
if not os.path.exists(configs.DATASET_EGOHANDS_DIR):
    os.mkdir(configs.DATASET_EGOHANDS_DIR)
#Create directory dataset/dataset_egohands/images/
if not os.path.exists(configs.DATASET_EGOHANDS_IMAGES_DIR):
    os.mkdir(configs.DATASET_EGOHANDS_IMAGES_DIR)
#Create directory dataset/dataset_egohands/annotations/
if not os.path.exists(configs.DATASET_EGOHANDS_ANNOTATIONS_DIR):
    os.mkdir(configs.DATASET_EGOHANDS_ANNOTATIONS_DIR)
#Create directory dataset/dataset_egohands/images/hand/
if not os.path.exists(configs.DATASET_EGOHANDS_IMAGES_HAND_DIR):
    os.mkdir(configs.DATASET_EGOHANDS_IMAGES_HAND_DIR)
#Create directory dataset/dataset_egohands/images/no_hand/
if not os.path.exists(configs.DATASET_EGOHANDS_IMAGES_NOHAND_DIR):
    os.mkdir(configs.DATASET_EGOHANDS_IMAGES_NOHAND_DIR)
#Create directory dataset/dataset_egohands/images/raw/
if not os.path.exists(configs.DATASET_EGOHANDS_IMAGES_RAW_DIR):
    os.mkdir(configs.DATASET_EGOHANDS_IMAGES_RAW_DIR)

#Create directory dataset/dataset/
if not os.path.exists(configs.DATASET_DATASET_DIR):
    os.mkdir(configs.DATASET_DATASET_DIR)
#Create directory dataset/dataset/train/
if not os.path.exists(configs.DATASET_DATASET_TRAIN_DIR):
    os.mkdir(configs.DATASET_DATASET_TRAIN_DIR)
#Create directory dataset/dataset/train/hand/
if not os.path.exists(configs.DATASET_DATASET_TRAIN_HAND_DIR):
    os.mkdir(configs.DATASET_DATASET_TRAIN_HAND_DIR)
#Create directory dataset/dataset/train/no_hand/
if not os.path.exists(configs.DATASET_DATASET_TRAIN_NOHAND_DIR):
    os.mkdir(configs.DATASET_DATASET_TRAIN_NOHAND_DIR)
#Create directory dataset/dataset/validation/
if not os.path.exists(configs.DATASET_DATASET_VALIDATION_DIR):
    os.mkdir(configs.DATASET_DATASET_VALIDATION_DIR)
#Create directory dataset/dataset/validation/hand/
if not os.path.exists(configs.DATASET_DATASET_VALIDATION_HAND_DIR):
    os.mkdir(configs.DATASET_DATASET_VALIDATION_HAND_DIR)
#Create directory dataset/dataset/validation/no_hand/
if not os.path.exists(configs.DATASET_DATASET_VALIDATION_NOHAND_DIR):
    os.mkdir(configs.DATASET_DATASET_VALIDATION_NOHAND_DIR)
#Create directory dataset/dataset/test/
if not os.path.exists(configs.DATASET_DATASET_TEST_DIR):
    os.mkdir(configs.DATASET_DATASET_TEST_DIR)
#Create directory dataset/dataset/test/hand/
if not os.path.exists(configs.DATASET_DATASET_TEST_HAND_DIR):
    os.mkdir(configs.DATASET_DATASET_TEST_HAND_DIR)
#Create directory dataset/dataset/test/no_hand/
if not os.path.exists(configs.DATASET_DATASET_TEST_NOHAND_DIR):
    os.mkdir(configs.DATASET_DATASET_TEST_NOHAND_DIR)


# -------------------------------------------------------------------------------
# MOVE DATASETS TO dataset/type_dataset/images/raw 
# where type_dataset = dataset_egohands or dataset_handsoverfaces
# -------------------------------------------------------------------------------

print("MOVING IMAGES FROM {} TO {} ".format(configs.LABELLED_SAMPLES_DIR,configs.DATASET_EGOHANDS_IMAGES_RAW_DIR))
helper.move_egohands_dataset(configs.LABELLED_SAMPLES_DIR,configs.DATASET_EGOHANDS_IMAGES_RAW_DIR, configs.FOLDERS_TO_IGNORE_EGOHANDS)

print("MOVING IMAGES FROM {} TO {} ".format(configs.IMAGES_ORIGINAL_SIZE_DIR,configs.DATASET_HANDSOVERFACES_IMAGES_RAW_DIR))
helper.move_handoverface_dataset(configs.IMAGES_ORIGINAL_SIZE_DIR,configs.DATASET_HANDSOVERFACES_IMAGES_RAW_DIR,configs.FILES_TO_IGNORE)

# -------------------------------------------------------------------------------
# MOVE ANNOTATIONS TO dataset/type_dataset/annotations/ 
# where type_dataset = dataset_egohands or dataset_handsoverfaces
# -------------------------------------------------------------------------------

print("MOVING TXTS FROM {} TO {} ".format(configs.LABELLED_SAMPLES_DIR,configs.DATASET_EGOHANDS_ANNOTATIONS_DIR))
helper.move_egohands_annoatations(configs.LABELLED_SAMPLES_DIR,configs.DATASET_EGOHANDS_ANNOTATIONS_DIR, configs.FOLDERS_TO_IGNORE_EGOHANDS)

#creation txt files for hand over face dataset
print("CONVERTING INTO TXTS FORMAT ANNOTATIONS OF HAND OVER FACE DATASET")
helper.save_annotations_txt_format_handsoverfaces()

print("MOVING TXTS FROM {} TO {}".format(configs.HANDSOVERFACES_HANDOVERFACE_ANNOTATIONS,configs.DATASET_HANDSOVERFACES_ANNOTATIONS_DIR))
helper.move_handoverface_annoatations(configs.HANDSOVERFACES_HANDOVERFACE_ANNOTATIONS,configs.DATASET_HANDSOVERFACES_ANNOTATIONS_DIR,configs.FILES_TO_IGNORE)

# -------------------------------------------------------------------------------
# CREATE POSITIVE AND NEGATIVE SAMPLES FOR THE TWO DATSETS
# -------------------------------------------------------------------------------

print("CREATING POSITIVE IMAGES EGOHANDS DATASET, SAVING TO {}".format(configs.DATASET_EGOHANDS_IMAGES_HAND_DIR))
helper.create_positive_egohandsdataset()

print("CREATING POSITIVE IMAGES HANDS OVER FACE DATASET, SAVING TO {}".format(configs.DATASET_HANDSOVERFACES_IMAGES_NOHAND_DIR))
helper.create_positive_handsoverfacesdatset()

print("CREATING NEGATIVE IMAGES EGOHANDS DATASET, SAVING TO {}".format(configs.DATASET_EGOHANDS_IMAGES_NOHAND_DIR))
helper.create_negative_egohandsdatset()

print("CREATING NEGATIVE IMAGES EGOHANDS DATASET, SAVING TO {}".format(configs.DATASET_HANDSOVERFACES_IMAGES_NOHAND_DIR))
helper.create_negative_handsoverfaces()

# -------------------------------------------------------------------------------
# CREATE DATASET FOR CNN
# -------------------------------------------------------------------------------

print("CREATING DATASET FOR CNN")
helper.creating_dataset_cnn()

# -------------------------------------------------------------------------------
# REMOVE NO MORE NECESSARY THINGS
# -------------------------------------------------------------------------------

print("REMOVING UNNECESSARY STUFF")
helper.remove_unnecessary_things()


