#ifndef DETECTOR_H
#define DETECTOR_H

//OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

//STL
#include <vector>
#include <tuple>

/*
*
* ------------ y
* |
* |
* |
* x
*/

class Detector 
{
public:
	
	/**
	* This function will read the images inside pathImages 
	* @param pathImages : The path were the images are
	*/
	void readImages(cv::String pathImages);

	/**
	* This function will return a vector of images where each image has inside of it the bounding boxes drawn
	*/
	std::vector<cv::Mat> detectHands();
	
	//Better if in the constructor??
	/**
	* This function will set the path of the model
	* @param : Path where the model is 
	*/
	void setModel(cv::String pathModel);

private:

	//FUNCTIONS

	/**
	* This function return given an image the bounding boxes of the detected hands inside of it
	* @param image : Image for which detecting the hands
	* @return : The list of bounding boxes where hands are supposed to be present
	*/
	std::vector<cv::Range> getBoudingBoxesDetections(cv::Mat image);

	/**
	* This function return the images of the gaussian pyramid for a given image
	* @param image : The image for which computing the pyramid of guassian images
	* @return : The pyramid of images for the one provided
	*/
	std::vector<cv::Mat> getGaussianPyramid(cv::Mat image);

	/**
	* This function return the list of bounding boxes where inside the image there is a hand
	* @param image : The image for which we need to search the hands
	* @param originalDimensions : The original dimensions of the image (rows,cols)
	* @return : The Bounding Boxes of the image specified in the orginal dimension coordinates 
	*/
	std::vector<cv::Rect> getHandsBoundingBoxes(cv::Mat image,std::tuple<int,int> orginalDimensions);

	/**
	* This function given and input an image, it transform an image of size (224,224) and normalize it
	* @param image : The image to be "tranformed"
	* @return : The tranformed image
	*/
	cv::Mat prepareImageForCNN(cv::Mat image);


	/**
	* This function given as input an image in the correct format, it will return if it is an hand or not
	* @return : True if it is an hand, false otherwise
	*/
	bool isHand(cv::Mat image);

	/**
	* This function convert (x,y) coordinates into a subsampled image into coordinates of original image
	* @param coordinatesToConvert : Coordinates to be converted
	* @param orginalDimensions : Original Dimensions of the image
	* @param currentDimensions : Dimensions of the subsampled image
	* @return : Coordinates converted 
	*/
	std::tuple<int, int> convertCoordinates(std::tuple<int, int> coordinatesToConvert, std::tuple<int, int> orginalDimensions, std::tuple<int, int> currentDimensions);



















	//FIELD MEMBER
	 
	//Images to process
	std::vector<cv::Mat> images;

	cv::String pathModel;


	//CONSTANTS
	
	//Scale for the gaussianpyramid
	const float SCALE_PYRAMID = 1.5;

	//Kernel from cv::pyrDown() function
	const cv::Mat KERNEL_PYRAMID = (cv::Mat_<float>(5, 5) << 1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256,
															4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
															6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256,
															4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
															1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256);

	//Window size
	const std::tuple<int, int> WINDOW_SIZE = std::make_tuple(112, 112);

	//Strides
	const int STRIDE_ROWS = 25;
	const int STRIDE_COLS = 25;

	//Input CNN
	const int WIDTH_INPUT_CNN = 224;
	const int HEIGHT_INPUT_CNN = 224;
};


#endif // !DETECTOR_H

