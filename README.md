# COMPUTER VISION PROJECT

In this project the goal is to detect and segment hands inside images.
In order to achieve this, the main idea was to use for the :

- detection : Sliding Window approach with a Resnet50V2 CNN pre trained able to distinguish between hands and not hands. Moreover, to avoid several overlappings a non maxima suppression is also implemented
- segmentation : A Resenet50 CNN with the a deeplab encoder/decoder architecture and, some post processing operations in order to segment properly hands inside the images

Overview :
* [C++ Code Compilation](#compiling-the-c-code)
* [Hand detection](#hand-detection)
    * [Dataset Creation](#creation-of-the-dataset)
    * [Training](#training-the-resnet50v2-to-recognize-hands)
    * [Model Conversion](#converting-the-h5-into-the-model-supported-by-opencv-library)
    * [Inference](#detect-hands-on-images)
* [Hand Segmentation](#hand-segmentation)
	* [Information before usage](#disclaimer)
	* [Dataset](#dataset)
	* [Training](#training)
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
- `-d` or `--detect` : to activativate the detection mode
- `-m` or `--model` : to specify the path of the model for detection
- `-i` or `--image` : to specify the path of the image for which detecting hands, default value : `../testset/rgb/01.jpg`
- `-a` or `--annotation` : to specify the path of the annotation for the image for which detecting hands
- Optional `--opd` : to specify the output path where to store the image with the bounding boxes drawn
- Optional `--opius` : to specify the output path where to store the ious results of the image

Notice that, at least one of the two optional parameters, must be included into the command line execution instruction.

Example of a command :

`./projectGroup05 -d -m="path_to_model" -i="path_to_image" -a="path_to_annotations" --opd="path_save_detection_result" --opious="path_save_ious"`

## Hand Segmentation

### Disclaimer

The segmentation process is carried out with the usage of pre-computed masks from a matlab model (*that we have developed ourselves and fine tuned it*) for the following reasons :

- The process of inference through the usage of opencv library require the usage of the conversion of the model (MATLAB), into one of the supported formats, as pointed out here : https://docs.opencv.org/3.4/d6/d0f/group__dnn.html#ga3b34fe7a29494a6a4295c169a7d32422 , so,
we have converted the model (.mat) file into the open neural network exchage format (.onnx), you can in fact, download the model in such a format from here : TODO LINK

- However, when we try to use the model with *cv::readNetFromOnnx* or *cv::readNet* and then we set input with *net.setInput(..)*, a we compute the output *net.forward(..)* a strange behaviour happens in particular :
	- The output computed with the usage of C++ code is different from the one computed with python.
	- In particular, using python, the ouput is correct while using C++ is not. The proof of such a strange result can be found in the `problems` directory in particular, *because we not inventing nothing*, just observe the difference of the contents between the two files : `problems/python/results.txt` and `problems/c++/source/results.txt`. 
	So, what we can conclude is that with python everything works while on C++ not. Moreover, here also the link of similar problem to ours one : https://discuss.tvm.apache.org/t/different-output-for-large-yolo-onnx-model-in-python-correct-and-c-incorrect/11537
	 
Notice that : the mentioned problem was encountered with opencv version 4.5.x compiled from source and also version 4.6.0 pre-compiled for C++, while, the opencv version of python used was 4.6.0

Therefore, the pre-computed masks for the testset (https://drive.google.com/drive/folders/1ORmMRRxfLHGLKgqHG-1PKx1ZUCIAJYoa?usp=sharing)can be downloaded from : TODO ADD LINK, as mentioned [here](#segment-hands-on-images)

### Dataset

The dataset used to train the model for segment hands can be downloaded from this link : TODO ADD LINK.

### Training

For training the model, you can use the script present inside `matlab_scripts/` TODO ADD NAME

### Segment Hands on images

First of all, in order to find out what are the hands inside an image you need to either :
- for each image that you what to segment, compute the output mask with the model previously trained
- if the testset is the one that can be downloaded from here : https://drive.google.com/drive/folders/1ORmMRRxfLHGLKgqHG-1PKx1ZUCIAJYoa?usp=sharing , then, just dowload the mask directly from : TODO ADD LINK

In order to segment hands on an image, you need to execute the `C++` code i.e. `./projectGroup05` with the following, possibiles parameters :
- `-s` or `--segment` : to activativate the segmentation mode
- `-i` or `--image` : to specify the path of the image for which detecting hands, default value : `../testset/rgb/01.jpg`
- `-a` or `--annotation` : to specify the path of the annotation for the image, i.e. the path to the ground truth mask
- `--bwm` : to specify the path where is the raw mask provided in output by the model
- Optional `--ops` : to specify the output path where to store the image with hands segmented drawn
- Optional `--oppa` : to specify the output path where to store the pixel accuracy results of the image
- Optional `--opbwm` : to specify the output path where to store the B&W mask

Notice that, at least one of the three optional parameters, must be included into the command line execution instruction.

Example of a command :
`./projectGroup05 -s -i="path_to_image" -a="path_to_mask" --bwm="path_bw_mask_model" --ops="path_save_segmentation_result" --oppa="path_save_pixelaccuracy" --opbwm="path_save_b&w_mask"`
