
//OPENCV

#include <opencv2/core/base.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


//STD
#include <iostream>
#include <fstream>

//MYLIB
#include "ImageDetection.h"
#include "UtilsHoughTransform.h"


using namespace std;
using namespace cv;
int main()
{
	// Read image
	Mat img = imread("myhand.jpg");

	Mat pattern = imread("myhand.jpg");


	UtilsHoughTransform util;

	util.generalizedHoughTransform(pattern, img);


	
}


void mainSimone()
{

	// Read image
	Mat img = imread("01.jpg");



	// Function that return an Image with bounding Box drawed
	Mat imgWithBoundingbox = ImageDetection::drawingBoundingBox(img, "01.txt");

	//Display image with bounding box
	imshow("keypoints", imgWithBoundingbox);
	waitKey(0);
}