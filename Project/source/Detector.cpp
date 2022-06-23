#include "../include/Detector.h"

void Detector::readImages(cv::String pathImages)
{
	//Read all the paths of the images
	std::vector<cv::String> pathSingleImages;
	cv::glob(pathImages, pathSingleImages);

	//Store the images inside the images vector
	for (int i = 0; i < pathSingleImages.size(); i++)
		images.push_back(cv::imread(pathSingleImages.at(i)));
}

std::vector<cv::Mat> Detector::detectHands()
{
	for (int i = 0; i < images.size(); i++)
	{
		getBoudingBoxesDetections(images.at(i));


	}


}

std::vector<cv::Range> Detector::getBoudingBoxesDetections(cv::Mat image)
{
	std::vector<cv::Mat> pyramid = getGaussianPyramid(image);

	return std::vector<cv::Range>();
}

std::vector<cv::Mat> Detector::getGaussianPyramid(cv::Mat image)
{
	std::vector<cv::Mat> pyramid;
	
	//Add the image as it is
	pyramid.push_back(image);

	cv::Mat temp = image.clone();
	while (true)
	{
		//Compute new rows and cols
		int rows = image.rows / SCALE_PYRAMID;
		int cols = image.cols / SCALE_PYRAMID;

		//Resize the image
		cv::Mat resized;
		cv::resize(temp, resized, cv::Size(),cv::INTER_CUBIC); //to check what I use previously

		//Apply Gaussian Smoothing

	}


	return pyramid;
}
