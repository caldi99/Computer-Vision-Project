#include "../include/Segmentator.h"
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

/**
* This file represent the Segmentator module
* @author : Daniela Cuza
*/

void Segmentator::segment_1(cv::String pathImage)
{
	// read the image
	Mat img = imread(pathImage, IMREAD_COLOR);

    //definition of variables
	Mat out_bf;
    Mat skin_region;
    Mat out;
    Mat gray;

	// step 1) apply bilateral filter
	bilateralFilter(img,out_bf, 5, 150, 150);
    

	// step 2) apply threshold
    cvtColor(out_bf, out_bf, COLOR_BGR2YCrCb);
    inRange(out_bf, Scalar(0, 133, 77), Scalar(255, 173, 127), skin_region);
    out_bf.copyTo(out, skin_region); //apply mask

	// step 3) find the contours 
    
    cvtColor(out, gray, COLOR_BGR2GRAY);

    Canny(gray, gray, 130, 180, 3);
    /// Find contours   
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    RNG rng(12345);
    findContours(gray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    /// Draw contours
    Mat drawing = Mat::zeros(gray.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
    }

    
    
	//show the result
	imshow("Input", img);
    imshow("Result window", drawing);

}