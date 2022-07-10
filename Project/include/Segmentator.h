#ifndef SEGMENTATOR_H
#define SEGMENTATOR_H

//OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

//STL
#include <vector>
#include <tuple>

//TODO THINK OF WHAT CAN BE PASSED AS REFERENCE

/**
* This class represent the Segmentator Module 
* @author : Daniela Cuza and Francesco Caldivezzi
*/
class Segmentator
{
public:

	/**
	* This function will read the images inside pathImages
	* @param pathImages : The path were the images are
	*/
	void segment_1(cv::String pathImage);

	/**
	* This function will read the image to segment inside pathImage
	* @param pathImage : The path were the image is
	*/
	void readImage(cv::String pathImage);

	//TODO : need to read it as a grayscale image?
	/**
	* This function will read the ground truth inside pathGroundTruth
	* @param pathGroundTruth : The path were the groundTruth mask is
	*/
	void readGroundTruth(cv::String pathGroundTruth);

	/**
	* This function will set the path of the model
	* @param : Path where the model is
	*/
	void setModel(cv::String pathModel);

	/**
	* This function will get the image read with readImage function
	* @return : The image read with readImage function
	*/
	cv::Mat getImage();

private:
	//FIELD MEMBER

	//Image to process
	std::tuple<cv::Mat, cv::String> image;

	//Path to the CNN model
	cv::String pathModel;

	//Ground Truth Mask
	cv::Mat groundTruth;
};
#endif // !SEGMENTATOR_H