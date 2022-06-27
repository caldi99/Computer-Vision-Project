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
	// **********  read the image ************** //
	Mat img = imread(pathImage, IMREAD_COLOR);


    
    // ************** definition of variables ************** //
	Mat out_bf;
    Mat skin_region;
    Mat src;


	// ********** step 1) apply bilateral filter ************** //
	bilateralFilter(img,out_bf, 5, 150, 150);

    
	// ********** step 2) apply threshold *************

    for (int i = 0; i < out_bf.rows; i++) {
        for (int j = 0; j < out_bf.cols; j++) {

            int R = out_bf.at<cv::Vec3b>(i, j)[0];
            int G = out_bf.at<cv::Vec3b>(i, j)[1];
            int B = out_bf.at<cv::Vec3b>(i, j)[2];
            int max_value;
            int min_value;

            // search max value among R, G, B
            if (R >= G && R >= B) {
                max_value = R;
            }
            else if (G >= R && G >= B) {
                max_value = G;
            }
            else {
                max_value = B;
            }

            // search min value among R, G, B
            if (R <= G && R <= B) {
                min_value = R;
            }
            else if (G <= R && G <= B) {
                min_value = G;
            }
            else {
                min_value = B;
            }

            if ((B > 95 && G > 40 && R > 20 && (max_value - min_value > 15) && abs(B - G) > 15 && B > G && B > R) || (B > 220 && G > 210 && R > 170 && abs(B - G) <= 15 && B > R && G > R)) {
                
                out_bf.at<cv::Vec3b>(i, j)[0] = R;
                out_bf.at<cv::Vec3b>(i, j)[1] = G;
                out_bf.at<cv::Vec3b>(i, j)[2] = B;
            }
            else {
                out_bf.at<cv::Vec3b>(i, j)[0] = 0;
                out_bf.at<cv::Vec3b>(i, j)[1] = 0;
                out_bf.at<cv::Vec3b>(i, j)[2] = 0;
            }
        }
    }
    
    
            
    cvtColor(out_bf, out_bf, COLOR_BGR2YCrCb); // covert from BGR to YCrCb
    inRange(out_bf, Scalar(0, 133, 77), Scalar(255, 173, 127), skin_region); //compute mask
    out_bf.copyTo(src, skin_region); //apply mask
    cvtColor(src, src, COLOR_YCrCb2BGR); // covert from YCrCb TO BGR
    
    
    imshow("Image after bilateral filter and threshold", src);

    
    
}