
#pragma warning (disable : 4996)

#include "../include/BagOfWords.h"

BagOfWords::BagOfWords(std::string pathDirectory, int numberFiles, std::string extension)
{
	//Read images, convert them and save
	readFiles(pathDirectory, numberFiles, extension);

	//Extract and save keypoints and descriptos
	cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> keypoints;

	for (cv::Mat img : trainingSet)
	{
		detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
		this->descriptors.push_back(descriptors);
		this->keypoints.push_back(keypoints);
	}
}

void BagOfWords::trainBagOfWords()
{
	//COMPUTE FEATURES FOR EACH TRAINING IMAGE
	cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> keypoints;
	for (cv::Mat img : trainingSet)
	{
		detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
		this->descriptors.push_back(descriptors);
		this->keypoints.push_back(keypoints);
	}

	//CLUSTER FEATURES
	cv::Mat labels;
	
	int attempts = 10;
	int K = 2;
	cv::kmeans(descriptors, K, labels, cv::TermCriteria(cv::TermCriteria::COUNT, 10000, 0.0001), attempts, cv::KMEANS_PP_CENTERS, centers);	
}

void BagOfWords::prediction(cv::Mat img)
{
	cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
	cv::Mat descriptors;
	std::vector<cv::KeyPoint> keypoints;
	detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

	// TODO: Finisci


}

void BagOfWords::readFiles(std::string pathDirectory, int numberFiles, std::string extension)
{
	char* pathFile = new char[pathDirectory.size() + extension.size() + 3];

	for (int i = 1; i <= numberFiles; i++)
	{
		sprintf(pathFile, pathDirectory.c_str());
		if (i < 10)
			sprintf(pathFile + pathDirectory.size(), "i0%d", i);
		else
			sprintf(pathFile + pathDirectory.size(), "i%d", i);
		sprintf(pathFile + pathDirectory.size() + 3, extension.c_str());

		trainingSet.push_back(cv::imread(pathFile,cv::IMREAD_GRAYSCALE));
	}
	delete pathFile;
}
