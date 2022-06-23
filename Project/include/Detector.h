#ifndef DETECTOR_H
#define DETECTOR_H

//OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//STL
#include <vector>


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


	//FIELD MEMBER
	std::vector<cv::Mat> images;

	const float SCALE_PYRAMID = 1.5;

	//Kernel from cv::pyrDown() function
	const cv::Mat KERNEL_PYRAMID = (cv::Mat_<float>(5, 5) << 1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256,
															4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
															6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256,
															4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256,
															1 / 256, 4 / 256, 6 / 256, 4 / 256, 1 / 256);
};


#endif // !DETECTOR_H

