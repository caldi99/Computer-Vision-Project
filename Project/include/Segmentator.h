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

/**
* This class represent the Segmentator Module 
* @author : Daniela Cuza, Simone D'antimo
*/
class Segmentator
{
public:

	/**
	* This function computes the true B&W  mask.
	* @return : The true B&W Mask
	*/
	cv::Mat getSegmentationMaskBW();

	/**
	* This function compute the image with segmented hands of different colors
	* param bwMask : The true BW mask
	* @return : The image with segmented hands of different colors
	*/
	cv::Mat getImageWithSegmentations(const cv::Mat& bwMask);

	/**
	* This function computes the Pixel Accuracies, given the B&W mask and save the result inside pathPixelAccuarcy
	* @param pathPixelAccuarcy : The path of the file where to save the Pixel Accuracies 
	* @param bwMask : The true B&W mask
	*/
	void savePixelAccuracies(cv::String pathPixelAccuarcy, const cv::Mat& bwMask);

	/**
	* This function save the image with segmented hands of different colors inside pathSegmentation, given the B&W mask
	* @param pathSegmentation : The path where to save the image with segmentations
	* @param bwMask : The true B&W mask
	*/
	void saveSegmentations(cv::String pathSegmentation, const cv::Mat& bwMask);

	/**
	* This function is used to save the B&W in pathSegmentationMaskBW
	* @param pathSegmentationMaskBW : Path where to save the B&W mask
	* @param bwMask : true B&W mask to save
	*/
	void saveSegmentationMaskBW(cv::String pathSegmentationMaskBW, const cv::Mat& bwMask);

	/**
	* This function will read the image to segment inside pathImage
	* @param pathImage : The path were the image is
	*/
	void readImage(cv::String pathImage);

	/**
	* This function will read the ground truth inside pathGroundTruth
	* @param pathGroundTruth : The path were the groundTruth mask is
	*/
	void readGroundTruth(cv::String pathGroundTruth);

	/**
	* This function will read the Raw B&W mask provided by the CNN model
	* @param : Path where Raw B&W mask is
	*/
	void readBWMaskRaw(cv::String pathBWMaskRaw);


private:	

	//FIELD MEMBER

	//Image to process
	std::tuple<cv::Mat, cv::String> image;

	//BW mask raw
	cv::Mat bwRawMask;

	//Ground Truth Mask
	cv::Mat groundTruth;

	//Struct for containing Data regaring pixel accuracy
	struct EvaluationData {

		//false positive
		float fp = 0;

		//false negative
		float fn = 0;

		//true positive
		float tp = 0;

		//true negative
		float tn = 0;

		//Recall = TruePositives / (TruePositives + FalseNegatives)
		float recall = 0.0f;

		//Precision = TruePositives / (TruePositives + FalsePositives)
		float precision = 0.0f;

		//Pixel Accuracy = (TrueNegatives + TruePositives) / (TruePositives + FalsePositives + TrueNegatives + TruePositives)
		float pixelAccuracy = 0.0f;
	};

	//FUNCTIONS
		
	/**
	* This function compute the Pixel Accuracy given the mask and ground thruth
	* @param bwMask : The mask with which compute the Pixel Accuracy
	* @return : The Pixel accuracy value
	*/
	EvaluationData computePixelAccuracy(const cv::Mat& bwMask);
		
	//CONSTANTS
	
	//255 value pixel
	const int HIGHEST_VALUE = 255;
};
#endif // !SEGMENTATOR_H