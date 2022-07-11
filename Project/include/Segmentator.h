#ifndef SEGMENTATOR_H
#define SEGMENTATOR_H

//OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

//STL
#include <vector>
#include <tuple>
#include <fstream>

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
	//void segment_1(cv::String pathImage);

	/**
	* This function computes the B&W mask.
	* @return : The B&W Mask
	*/
	cv::Mat getSegmentationMaskBW();

	/**
	* This function compute the image with segmented hands of different colors
	* param bwMask : The BW mask
	* @return : The image with segmented hands of different colors
	*/
	cv::Mat getImageWithSegmentations(cv::Mat bwMask);

	/**
	* This function computes the Pixel Accuracies, given the B&W mask and save the result inside outputFile
	* @param outputFile : The path of the file where to save the Pixel Accuracies 
	* @param bwMask : The B&W mask
	*/
	void savePixelAccuracies(cv::String outputFile, cv::Mat bwMask);

	/**
	* This function save the image with segmented hands of different colors inside ouput, given the B&W mask
	* @param outputPath : The path where to save the image with segmentations
	* @param bwMask : The B&W mask
	*/
	void saveSegmentations(cv::String outputPath, cv::Mat bwMask);

	/**
	* This function is used to save the B&W in outputFile
	* @param outputFile : Path where to save the B&W mask
	* @param bwMask : B&W mask to save
	*/
	void saveSegmentationMaskBW(cv::String outputFile, cv::Mat bwMask);

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


private:	
	//FUNCTIONS
	
	/**
	* This function converts the output of the CNN to a B&W mask
	* @param outputCNN : The ouput of the CNN
	* @return : The B&W mask
	*/
	cv::Mat convertOutputCNNToBWMask(cv::Mat outputCNN);
	
	
	
	
	//FIELD MEMBER

	//Image to process
	std::tuple<cv::Mat, cv::String> image;

	//Path to the CNN model
	cv::String pathModel;

	//Ground Truth Mask
	cv::Mat groundTruth;

	


	//CONSTANTS
	
	//Input CNN
	const int WIDTH_INPUT_CNN = 224;
	const int HEIGHT_INPUT_CNN = 224;

	//Threshold CNN
	const float THRESHOLD_CNN = 0.5;

};
#endif // !SEGMENTATOR_H