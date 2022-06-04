#include "HandDetector.h"

HandDetector::HandDetector(std::string pathPositiveImages, std::string pathNegativeImages, int detectorWidth, int detectorHeigth)
{
	//Load positive and negative images
	positiveImages = loadImages(pathPositiveImages);
	negativeImages = loadImages(pathNegativeImages);
	
	this->detectorWidth = detectorWidth;
	this->detectorHeigth = detectorHeigth;

	//Add labels +1 to positive images, -1 to negative ones
	labels.assign(positiveImages.size(), +1);
	labels.insert(labels.end(), negativeImages.size(), -1);

	//Create the SVM
	svm = cv::ml::SVM::create();
}

std::vector<cv::Mat> HandDetector::loadImages(const cv::String& directoryName)
{
	std::vector<cv::Mat> images;
	std::vector<cv::String> files;

	cv::glob(directoryName, files);

	for (int i = 0; i < files.size(); i++)
		images.push_back(cv::imread(files.at(i)));

	return images;
}

cv::Mat HandDetector::convertToMachineLeaning(const std::vector<cv::Mat>& samples)
{
	int rows = samples.size();
	int cols = std::max(samples.at(0).cols, samples.at(0).rows);

	cv::Mat transposed(1, cols, CV_32FC1);
	cv::Mat trainingData(rows, cols, CV_32FC1);
	
	for (int i = 0; i < samples.size(); i++)
	{
		if (samples.at(i).cols == 1)
		{
			cv::transpose(samples.at(i), transposed);
			transposed.copyTo(trainingData.row(i));
		}
		else if (samples.at(i).rows == 1)
			samples.at(i).copyTo(trainingData.row(i));
	}
	return trainingData;
}

std::vector<cv::Mat> HandDetector::computeHOGs(const cv::Size windowSize, const std::vector<cv::Mat>& imageList)
{
	cv::HOGDescriptor descriptor;
	std::vector<cv::Mat> gradientImages;
	cv::Mat grayImage;
	std::vector<float> descriptors;

	descriptor.winSize = windowSize;

	cv::Rect rect(0, 0, windowSize.width, windowSize.height);
	rect.x += (imageList.at(0).cols - rect.width) / 2;
	rect.y += (imageList.at(0).rows - rect.height) / 2;
	
	for (int i = 0; i < imageList.size(); i++)
	{
		cv::cvtColor(imageList.at(i), grayImage, cv::COLOR_BGR2GRAY);
		descriptor.compute(grayImage, descriptors, cv::Size(8, 8), cv::Size(0, 0)); //1st size stide, 2nd padding
		gradientImages.push_back(cv::Mat(descriptors).clone()); //really necessary .clone()??
	}

	return gradientImages;
}

void HandDetector::trainSVM(cv::Mat trainingSet,float coef0, int degree, cv::TermCriteria terminationCriteria, 
	int gamma, cv::ml::SVM::KernelTypes kernelType, float Nu, float P, float C, cv::ml::SVM::Types type)
{
	svm->setCoef0(coef0);
	svm->setDegree(degree);
	svm->setTermCriteria(terminationCriteria);
	svm->setGamma(gamma);
	svm->setKernel(kernelType);
	svm->setNu(Nu);
	svm->setP(P); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(C); // From paper, soft classifier
	svm->setType(type); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	


	
	svm->train(trainingSet, cv::ml::ROW_SAMPLE, cv::Mat(labels));

}



