#include "BagOfWords.h"

BagOfWords::BagOfWords(std::string pathDataset)
{



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
	cv::Mat centers;
	int attempts = 10;
	int K = 2;
	cv::kmeans(descriptors, K, labels, cv::TermCriteria(cv::TermCriteria::COUNT, 10000, 0.0001), attempts, cv::KMEANS_PP_CENTERS, centers);
	
	//LABEL EACH CLUSTER WITH IMAGES THAT HAVE FEATURES IN THAT CLUSTER

	

}


