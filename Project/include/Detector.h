#ifndef DETECTOR_H
#define DETECTOR_H

//OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

//STL
#include <vector>
#include <tuple>
#include <map>
#include <iostream>
#include <fstream>

//MYLIB
#include "Utils.h"

//TODO THINK OF WHAT CAN BE PASSED AS REFERENCE, AND WHAT AS CONST

/**
* This class represent the Detector Module "definitions"
* @author : Francesco Caldivezzi
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
	* This function will read the ground truth inside pathGroundTruths
	* @param pathGroundTruth : The path were the groundTruths are
	*/
	void readGroundTruth(cv::String pathGroundTruths);

	//TODO SPLIT FOR PATH.JPG
	/**
	* This function will return the bounding boxes for a given image
	* @param nameImage : The filename of the image for which we need to detect things
	* @return : Bounding Boxes of the hands detected
	*/
	std::vector<cv::Rect> detectHands(cv::String nameImage);
	
	//Better if in the constructor??
	/**
	* This function will set the path of the model
	* @param : Path where the model is 
	*/
	void setModel(cv::String pathModel);

	//TODO FINISH HERE, COMPUTATION FOR ALL IMAGES OF THE IOU in a file
	float getIntersectionOverUnion(cv::String nameImage, std::vector<cv::Rect> detections);

private:

	//FUNCTIONS

	/**
	* This function return given an image the bounding boxes of the detected hands inside of it
	* @param image : Image for which detecting the hands
	* @return : The list of bounding boxes where hands are supposed to be present
	*/
	std::vector<cv::Rect> getBoudingBoxesDetections(cv::Mat image);

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
	* @param positionPyramid : The position of the image in the pyramid
	* @param probabilities : The probabilities returned by the Network
	* @return : The Bounding Boxes of the image specified in the orginal dimension coordinates 
	*/
	std::vector<cv::Rect> getHandsBoundingBoxes(cv::Mat image,std::tuple<int,int> orginalDimensions,int positionPyramid, std::vector<float>& probabilities);

	/**
	* This function given and input an image, it transform an image of size (224,224) and normalize it
	* @param image : The image to be "tranformed"
	* @return : The tranformed image
	*/
	cv::Mat prepareImageForCNN(cv::Mat image);

	/**
	* This function given as input an image in the correct format, it will return if it is an hand or not
	* @return : True if it is an hand, false otherwise and the output of the network
	*/
	std::tuple<bool,float> isHand(cv::Mat image);

	/**
	* This function convert (x,y) coordinates into a subsampled image into coordinates of original image
	* @param coordinatesToConvert : Coordinates to be converted provided as (x,y)
	* @param orginalDimensions : Original Dimensions of the image provided as (rangey,rangex) = (heigth,width) = (rows,cols)
	* @param currentDimensions : Dimensions of the subsampled image provided as (rangey,rangex) = (heigth,width) = (rows,cols)
	* @return : Coordinates converted 
	*/
	std::tuple<int, int> convertCoordinates(std::tuple<int, int> coordinatesToConvert, std::tuple<int, int> orginalDimensions, std::tuple<int, int> currentDimensions);

	/**
	* This function given the detections as input, return the non maxima suppression of them
	* @param boxes : The detections for which appling Non Maxima Suppression
	* @param probabilities : List of probabilities if provided
	* @return : The result of Non Maxima Suppression
	*/
	std::vector<cv::Rect> nonMaximaSuppression(std::vector<cv::Rect> boxes, std::vector<float> probabilities = std::vector<float>());

	/**
	* This function convert a vector of rectangles specified with integer values into a rectangle that uses float values
	* @param boxes : The Bounding Boxes specified in integer coordinates
	* @return : The corresponding Bounding Boxes with float coordinates
	*/
	std::vector<cv::Rect2f> convertBoxesToFloatCoordinates(std::vector<cv::Rect> boxes);
	
	/**
	* This function convert a vector of rectangles specified with float values into a rectangle that uses int values
	* @param boxesFloat : The Bounding Boxes specified in float coordinates
	* @return : The corresponding Bounding Boxes with int coordinates
	*/
	std::vector<cv::Rect> convertBoxesToIntCoordinates(std::vector<cv::Rect2f> boxesFloat);
	
	/**
	* This function computes the intersecion over union
	* @param box1 : The first box
	* @param box2 : The second box
	* @return : The Intersection over union
	*/
	float intersectionOverUnion(cv::Rect box1, cv::Rect box2);

	/**
	* This function computes the intersecion over union element wise between boxes and box
	* @param boxes : The list of boxes box
	* @param box : A box
	* @return : The Element Wise Intersection over union
	*/
	std::vector<float> intersectionOverUnionElementWise(std::vector<cv::Rect>& boxes, cv::Rect& box);

	/**
	* This function creates a list of Boxes given lists of x1s, y1s, ws, hs
	* @param x1s : The list of x1s
	* @param y1s : The list of y1s
	* @param ws : The list of ws
	* @param hs : The list of hs
	* @return : The corresponding rectangles
	*/
	std::vector<cv::Rect> createListBoxes(std::vector<float>& x1s, std::vector<float>& y1s, std::vector<float>& ws, std::vector<float>& hs);

	//FIELD MEMBER
	 
	//Images to process
	std::map<cv::String,cv::Mat> images;

	//Ground truth images
	std::map<cv::String, std::vector<cv::Rect>> groundTruths;

	cv::String pathModel;


	//CONSTANTS
	
	//Scale for the gaussianpyramid
	const float SCALE_PYRAMID = 1.5;

	//Kernel from cv::pyrDown() function
	const cv::Mat KERNEL_PYRAMID = (cv::Mat_<float>(5, 5) << 1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
															4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
															6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
															4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
															1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0);

	//Window size
	const std::tuple<int, int> INITIAL_WINDOW_SIZE = std::make_tuple(224, 224);

	//Strides
	const float STRIDE_ROWS_FACTOR = 0.5;
	const float STRIDE_COLS_FACTOR = 0.5;

	//Input CNN
	const int WIDTH_INPUT_CNN = 224;
	const int HEIGHT_INPUT_CNN = 224;

	//Threshold used to understand if a blob is an image or not
	const float THRESHOLD_DETECTION = 0.5;

	//Threshold used to understand how much two overlapping regions overlap each other
	const float THRESHOLD_OVERLAPPING = 0.85;
};


#endif // !DETECTOR_H


