#include "../include/PreProcSegmentator.h"
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

/**
* This file represent the preprocessing for the segmentation 
* @author : Simone D'Antimo
*/

 void PreProcSegmentator::getPreFiltered(Mat& src, Mat& binaryDst, Mat& laplacianDst) {
    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
    Mat mask;
    inRange(src, Scalar(255, 255, 255), Scalar(255, 255, 255), mask);
    src.setTo(Scalar(0, 0, 0), mask);
    // Show output image
    imshow("Black Background Image", src);
    // Create a kernel that we will use to sharpen our image
    Mat kernel = (Mat_<float>(3, 3) <<
        1, 1, 1,
        1, -8, 1,
        1, 1, 1);

    // As kernel we use an approximation of second derivative
    // possible negative number will be truncated

    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
  //  imshow("New Sharped Image", imgResult);
 //   imshow("New Laplacian Image", imgLaplacian);


    // Create binary image from source image
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    //  imshow("Binary Image", bw);
    laplacianDst = imgLaplacian;
    binaryDst = bw;


}

