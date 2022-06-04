#ifndef HANDDETECTOR_H
#define HANDDETECTOR_H


//STL
#include <vector>


//OPENCV
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>

/// <summary>
/// </summary>
class HandDetector
{
public:
	

	//TODO ALL POSITIVE IMAGES MUST BE OF SAME SIZE

	/// <summary>
	/// This Constructor creates a new Detector
	/// </summary>
	/// <param name="pathPositiveImages">Path of the folder where positive images are present</param>
	/// <param name="pathNegativeImages">Path of the folder where negative images are present</param>
	/// <param name="detectorWidth">Detector Width</param>
	/// <param name="detectorHeigth">Detector Heigth</param>
	HandDetector(cv::String pathPositiveImages, cv::String pathNegativeImages, int detectorWidth, int detectorHeigth);
	



private:

	std::vector<cv::Mat> positiveImages;
	std::vector<cv::Mat> negativeImages;
	int detectorWidth;
	int detectorHeigth;	
	std::vector<int> labels;
	cv::Ptr<cv::ml::SVM> svm;


	/// <summary>
	/// This Function will return the images inside the directory specified as parameter
	/// </summary>
	/// <param name="directoryName">The directory where images are stored</param>
	/// <returns>Images Inside the directory</returns>
	std::vector <cv::Mat> loadImages(const cv::String& directoryName);
	
	/// <summary>
	/// Convert a list of sample images to be used by OpenCV Machine Learning algorithms
	/// </summary>
	/// <param name="samples">The samples to be converted</param>
	/// <returns>A matrix of size (#sample x max(#cols,#rows), in 32FC1</returns>
	cv::Mat convertToMachineLeaning(const std::vector<cv::Mat>& samples);

	/// <summary>
	/// This function computes the histogram of gradients of the given images
	/// </summary>
	/// <param name="windowSize">Window size</param>
	/// <param name="imageList">Images for which computing the list of histogram gradients</param>
	/// <returns>List of histogram of gradients</returns>
	std::vector <cv::Mat> computeHOGs(const cv::Size windowSize, const std::vector<cv::Mat>& imageList);

	/// <summary>
	/// This function will train the SVM with the given parameters
	/// </summary>
	/// <param name="trainingSet">Training Set for the SVM</param>
	/// <param name="coef0">Coef0 parameter of the SVM</param>
	/// <param name="degree">Degree parameter of the SVM</param>
	/// <param name="terminationCriteria">TerminationCriteria of the SVM</param>
	/// <param name="gamma">Gamma of the SVM</param>
	/// <param name="kernelType">Kernel Type of the SVM</param>
	/// <param name="Nu">Nu of the SVM</param>
	/// <param name="P">P of the SVM</param>
	/// <param name="C">C of the SVM</param>
	/// <param name="type">Type of the SVM</param>
	void trainSVM(cv::Mat trainingSet,float coef0, int degree, cv::TermCriteria terminationCriteria,
		int gamma, cv::ml::SVM::KernelTypes kernelType,float Nu,float P,float C, cv::ml::SVM::Types type);

};


#endif // !HANDDETECTOR_H

