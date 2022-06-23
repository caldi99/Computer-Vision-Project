#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/xfeatures2d.hpp>

class Detector
{
public:
	Detector();
	//inference
	bool loadDictionary(std::string dictionary_path = "dictionary.yml");
	bool loadSVM(std::string svm_path = "svm.yml");
	bool detect(cv::Mat input);
	bool detect(cv::Mat input, cv::Mat mask);
	double evaluate(cv::Mat mask, std::string gt_path);

private:
	cv::Ptr<cv::ml::SVM> svm;
	cv::Mat dictionary;

	cv::Ptr<cv::DescriptorMatcher> matcher;
	cv::Ptr<cv::DescriptorExtractor> extractor;
	cv::Ptr<cv::BOWImgDescriptorExtractor> bow_extractor;
};
