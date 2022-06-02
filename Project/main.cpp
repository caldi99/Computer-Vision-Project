
#include <iostream>
#include <opencv2/core/base.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

#include "include/ImageDetection.h"

using namespace std;
using namespace cv;
int main()
{


	
	// Read image
	//Mat img = imread("01.jpg");

	Mat img = imread("patten.jpg");
	
	// Function that return an Image with bounding Box drawed
	Mat imgWithBoundingbox = ImageDetection::drawingBoundingBox(img, "01.txt");

	//Display image with bounding box
	imshow("keypoints", imgWithBoundingbox);
	waitKey(0);
	return 0;
}