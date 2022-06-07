
//OPENCV
#include <opencv2/core/base.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

//STD
#include <iostream>
#include <fstream>

//MYLIB
#include "../include/ImageDetection.h"
#include "../include/UtilsHoughTransform.h"
#include "../include/ConvexHull.h"
#include "../include/HandDetector.h"

void mainFrancesco();
void mainSimone();
void mainTraining();

using namespace std;
using namespace cv;


int mainF()
{
	
	//mainFrancesco();

    mainTraining();
}

void mainTraining()
{
    /*Mat img = imread("../Data/HOG-object-detection-master/hand-pos/filename002.jpg");
    namedWindow("pippo");
    imshow("pippo", img);
    waitKey();*/

    HandDetector detector("../Data/HOG-object-detection-master/hand-pos/", "../Data/HOG-object-detection-master/eggs-neg/", 24, 24);
    detector.trainHandDetector();
    vector<Mat> img = detector.testHandDetector("../Data/HOG-object-detection-master/mani/");

}

void mainFrancesco()
{
	//Mat img = imread("01.jpg");
	
	//Mat gray;
	//cvtColor(img, gray, COLOR_BGR2GRAY);

	//imshow("pippo", gray);
	//waitKey();
	//destroyAllWindows();

	UtilsConvexHull util;
	//util.computeHull(gray);

	/*// Read image
	Mat img = imread("../images/myhand.jpg");

	Mat pattern = imread("../myhand.jpg");


	UtilsHoughTransform util;

	util.generalizedHoughTransform(pattern, img);*/


    Mat frame, roi, hsv_roi, mask;
    // take first frame of the video
    frame = imread("01.jpg");
    // setup initial location of window
    Rect track_window(200, 200, 200, 200); // simply hardcoded the values
    // set up the ROI for tracking
    roi = frame(track_window);
    cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
    inRange(hsv_roi, Scalar(0, 60, 32), Scalar(180, 255, 255), mask);
    float range_[] = { 0, 180 };
    const float* range[] = { range_ };
    Mat roi_hist;
    int histSize[] = { 180 };
    int channels[] = { 0 };
    calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
    normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

    // Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);
    while (true) {
        Mat hsv, dst;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        calcBackProject(&hsv, 1, channels, roi_hist, dst, range);
        // apply meanshift to get the new location
        meanShift(dst, track_window, term_crit);
        // Draw it on image
        rectangle(frame, track_window, 255, 2);
        imshow("img2", frame);
        int keyboard = waitKey(1);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }

    
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

void main()
{
    // main Daniela

    // Read image
    Mat img = imread("01.jpg");



    // Function that return an Image with bounding Box drawed
    Mat imgWithBoundingbox = ImageDet::drawingNegBoundingBoxx(img, "01.txt");

    //Display image with bounding box
    imshow("keypoints", imgWithBoundingbox);
    waitKey(0);
}