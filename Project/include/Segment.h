#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#define GRID_SCALE 0.05

class Segment
{
public:
	static std::vector<cv::Mat> segments;

	//preprocessing methods
	static void preprocess_BF(cv::Mat src, cv::Mat& output); //applies a bilateral filter
	static void preprocess_MS(cv::Mat src, cv::Mat& output, int variant = 0); //applies mean shift color quantization
	static void preprocess_LA(cv::Mat src, cv::Mat& output); //applies a Laplacian filter

	static void segment(cv::Mat src, cv::Mat& segment_map, bool show_steps = false); //segmentation via "intelligent" floodfill
};
