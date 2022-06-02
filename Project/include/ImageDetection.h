#ifndef IMAGEDETECTION_H
#define IMAGEDETECTION_H
#define COLORBB cv::Scalar(0,255,0)

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string.h>
#include <fstream>

class ImageDetection
{
	public:
		static cv::Mat drawingBoundingBox(cv::Mat& image, std::string filename);
};

#endif

