#include "../include/Segmentator.h"
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

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
	// step 1) apply bilateral filter
	bilateralFilter(img,out_bf, 5, 75, 75);

	// step 2) apply find contour
    //Prepare the image for findContours
    cvtColor(out_bf, out_bf, COLOR_BGR2YCrCb);

    // threshold

    inRange(out_bf, Scalar(0, 133, 77), Scalar(255, 173, 127), skin_region);

    out_bf.copyTo(out, skin_region);
    //Mat contourOutput = out_bf.clone();
    //findContours(contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    /*
    //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat contourOutput = out_bf.clone();
    cv::findContours(contourOutput, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    //Draw the contours
    cv::Mat contourImage(out_bf.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Scalar colors[3];
    colors[0] = cv::Scalar(255, 0, 0);
    colors[1] = cv::Scalar(0, 255, 0);
    colors[2] = cv::Scalar(0, 0, 255);
    for (size_t idx = 0; idx < contours.size(); idx++) {
        cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
    }
	// step 3) apply thershold
    */
	//show the result
	imshow("Input", img);
	imshow("Output", out);
}