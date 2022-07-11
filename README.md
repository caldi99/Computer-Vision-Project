# COMPUTER VISION PROJECT


In this project the goal is to detect and segment hands inside images.
In order to achieve this, the main idea was to use for the :

- detection : Sliding Window approach with a Resnet50V2 CNN pre trained able to distinguish between hands and not hands. Moreover, to avoid several overlappings a non maxima suppression is also implemented
- segmentation : TODO

Overview :
* [C++ Code Compilation](#compiling-the-c-code)
* [Hand detection](#hand-detection)
    * [Dataset Creation](#creation-of-the-dataset)
    * [Training](#training-the-resnet50v2-to-recognize-hands)
    * [Model Conversion](#converting-the-h5-into-the-model-supported-by-opencv-library)
    * [Inference](#detect-hands-on-images)
* [Hand Segmentation](#hand-segmentation)
	* TODO: Dataset creation, Training, Model
	* 
	* [Inference](#segment-hands-on-images)

## Compiling the C++ Code

In order to compile the code, you need to follow the upcoming steps :
- Go inside the `Project` directory i.e. `cd Project`
- Create a directory called `build` i.e. `mkdir build`
- Execute the following two commands `cmake ..` and `cmake --build .`

## Hand Detection

### Creation of the dataset

To create the dataset in order to train our model we need to follow the steps listed below:
- Dowload the EgoHands Dataset from : http://vision.soic.indiana.edu/egohands_files/egohands_data.zip
- Dowload the hand_over_face Dataset from : https://drive.google.com/file/d/1hHUvINGICvOGcaDgA5zMbzAIUv7ewDd3
- Dowload the TestSet from : https://drive.google.com/drive/folders/1ORmMRRxfLHGLKgqHG-1PKx1ZUCIAJYoa?usp=sharing
- Put all of those files in the root directory 
- Execute `python_scripts/build.py` i.e. : `python python_scripts/build.py` or `cd python_scripts` and then `python build.py` and follow the *instructions* you will asked to execute some matlab code, please do it!!
- Go inside the directory `dataset/dataset/` and type `ctrl + a` on the keyboad, and rigth click and compress as a `.zip` file.

Notice that, you can skip the above steps and download immediatly the `dataset.zip` file from : https://drive.google.com/file/d/1AxwsNnBCtxB2LLJ1q_N-YbTov9zbgKMh/view?usp=sharing

To create the dataset, it was done the following split :
 - 10% Test Set
 - 90% Traing Set and Validation Set in particular :
	- 75 % Training Set
	- 25 % Validation Set

### Training the Resnet50V2 to recognize hands

In order to train the CNN you need to follow the following steps : 

- if we want to train locally the CNN we need to follow the steps lists below :
	- First, you need to make sure that all the dependancies needed for the training process are satified. To do so, you need to check that in your enviroment (conda / pip), the following packages are installed :
	  - tensorflow (better if gpu version)
	  - opencv
	  - scikitlearn
	  - numpy
	  - matplotlib
	  - imutils
	- If all the packages are present, then, create the directory `python_scripts/dataset/` and extract inside of it the `dataset.zip` file
	- Next, execute `python_scripts/fine_tune_cnn.py`, a file `model.h5` will appear inside `python_scripts/`

- If we want to use google colab for the training process we need to follow the steps lists below : 
	- Upload the file `dataset.zip` previusly created (or downloaded) in your private google drive and place it in a directory called `CV`
	- Then, by using this link : [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a7S4M3odeVacq8i811q8c8Aacna47eTq?usp=sharing)  open the notebook that must be executed
	- After, opening the notebook upload on the root of the colab enviroment (`/content/` folder) the following files/folders :
		- `python_scripts/fine_tune_cnn.py` (file)
		- `python_scripts/config/` (folder and its content)
	- Moreover, execute all the code in the cells of the notebook **except the last one**
	- Finally a `model.h5` file will appear on the root of the colab enviroment, well, download such file

Notice that, you can skip the training process and download immediatly the `model.h5` file, from : https://drive.google.com/file/d/1vm2T1bqheUVgpB0QdJYq9mGdIplQ6f4H/view?usp=sharing

### Converting the .h5 into the model supported by OpenCV library

The last step is to convert the `model.h5` into .pb file in order to be able to use it in OpenCV. To do so, you need to follow the below steps :

- if you want to do it locally :
	- Optional : place the `model.h5` file under `python_scripts/` (notice that, this step must be done only if you trained your model with google colab)
	- Execute `python_scripts/convert_model_to_opencv.py` and then inside `python_scripts/model/` there will be the file `model.pb` that can be later used to do inference

- if you want to do it with google colab then:
	- Optional : By using this link : [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a7S4M3odeVacq8i811q8c8Aacna47eTq?usp=sharing) open the notebook that must be executed (if not already opened)
	- Optional : upload the `model.h5` file in the root of the google colab enviroment (notice that this must be done if the training process was done locally)
	- Upload in the root of the of the google colab enviroment :
		- `python_scripts/convert_model_to_opencv.py` (file)
		- Optional : `python_scripts/config/` (folder and its content) (if not already done)
	- Run the last cell of the notebook, and finally download the file `model/model.pb`

Notice that, you can skip this process and download immediatly the `model.pb` file from : https://drive.google.com/file/d/12nhBovdFL4O7X1d0FZbZOiSmgfYZnWmj/view?usp=sharing


### Detect Hands on images

In order to detect hands on an image, you need to execute the `C++` code i.e. `./projectGroup05` with the following, possibiles parameters :

- `-m` or `--model` : to specify the path of the model for detection
- `-i` or `--image` : to specify the path of the image for which detecting hands, default value : `../testset/rgb/01.jpg`
- `-a` or `--annotation` : to specify the path of the annotation for the image for which detecting hands
- `-d` or `--detect` : to activativate the detection mode
- Optional `--opd` : to specify the output path where to store the image with the bounding boxes drawn
- Optional `--opius` : to specify the output path where to store the ious results of the image

Notice that, at least one of the two optional parameters, must be included into the command line execution instruction.

Example of a command :

`./projectGroup05 -m="path_to_model" -i="path_to_image" -a="path_to_annotations" -d --opd="path_save_detection_result" --opious="path_save_ious"`

## Hand Segmentation
TODO COMPLETE

### Segment Hands on images

In order to segment hands on an image, you need to execute the `C++` code i.e. `./projectGroup05` with the following, possibiles parameters :
- `-m` or `--model` : to specify the path of the model for segmentation
- `-i` or `--image` : to specify the path of the image for which detecting hands, default value : `../testset/rgb/01.jpg`
- `-a` or `--annotation` : to specify the path of the annotation for the image, i.e. the path to the ground truth mask
- `-s` or `--segment` : to activativate the segmentation mode
- Optional `--ops` : to specify the output path where to store the image with hands segmented drawn
- Optional `--oppa` : to specify the output path where to store the pixel accuracy results of the image
- Optional `--opbwm` : to specify the output path where to store the B&W mask
Notice that, at least one of the three optional parameters, must be included into the command line execution instruction.

Example of a command :
`./projectGroup05 -m="path_to_model" -i="path_to_image" -a="path_to_masks" -s --ops="path_save_segmentation_result" --oppa="path_save_pixelaccuracy" --opbwm="path_save_b&w_mask"`




