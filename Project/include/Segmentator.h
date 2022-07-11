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
	* @return : The image with segmented hands of different colors
	*/
	cv::Mat getImageWithSegmentations();

	/**
	* This function computes the Pixel Accuracies, given the B&W mask and save the result inside outputFile
	* @param outputFile : The path of the file where to save the Pixel Accuracies 
	* @param bwMask : The B&W mask
	*/
	void savePixelAccuracies(cv::String outputFile, cv::Mat bwMask);

	/**
	* This function save the image with segmented hands of different colors inside ouput, given the B&W mask
	* @param ouput : The path where to save the image with segmentations
	* @param bwMask : The B&W mask
	*/
	void saveSegmentations(cv::String output, cv::Mat bwMask);

	//TODO:THINK IF NECESSARY
	void saveSegmentationMaskBW();

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
	//FUNCTIONS
	
	
	
	
	
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


};
#endif // !SEGMENTATOR_H