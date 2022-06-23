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

#define TRAIN_STRIDE 1 //skip over some images?
#define TRAIN_OFFSET 0 //if stride >1, apply some offset?
#define SVM_STRIDE 1 //skip over some images?
#define SVM_OFFSET 0 //if stride >1, apply some offset?
#define CLASS_COUNT 1000 //classes of features in the dictionary


class Trainer
{
public:
	Trainer();
	bool loadDatasetPaths(std::string dataset_folder); //for training a dictionary
	bool loadDictionary(std::string dictionary_path = "dictionary.yml"); //for training a SVM
	bool trainDictionary(std::string output_path = "dictionary.yml");
	bool trainSVM(std::string output_path = "svm.yml");

private:
	const std::string IMG_FOLDER = "IMAGES";
	const std::string IMG_EXTENSION = ".jpg";
	const std::string GT_FOLDER = "LABELS_TXT";
	const std::string FILE_PREFIX = "image";

	std::string dataset_root_path;
	std::vector<cv::String> dataset_paths;

	cv::Ptr<cv::ml::SVM> svm;
	cv::Mat dictionary;

	cv::Ptr<cv::DescriptorMatcher> matcher;
	cv::Ptr<cv::DescriptorExtractor> extractor;
	cv::Ptr<cv::BOWImgDescriptorExtractor> bow_extractor;
};
