
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
#include "../include/Trainer.h"
#include "../include/Segment.h"
#include "../include/Detector.h"

void mainFrancesco();
void mainSimone();
void mainTraining();
int mainTrainer(int argc, char** argv);

using namespace std;
using namespace cv;

enum MODE {
	DETECT, BUILD_DICTIONARY, BUILD_SVM, BUILD_ALL
};

int main2(int argc, char** argv)
{
	
	//mainFrancesco();
	mainTrainer(argc, argv);
    //mainTraining();
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

void mainDaniela()
{
    // main Daniela

    // Read image
    Mat img = imread("01.jpg");



    // Function that return an Image with bounding Box drawed
    //Mat imgWithBoundingbox = ImageDet::drawingNegBoundingBoxx(img, "01.txt");

    //Display image with bounding box
    //imshow("keypoints", imgWithBoundingbox);
    waitKey(0);
}



int mainTrainer(int argc, char** argv) 
{
	const cv::String keys =
		"{help h usage ? |               | print this message   }"
		"{i detect       |               | run inference/detection }"
		"{d train_dict   |               | train dictionary }"
		"{s train_svm    |               | train SVM }"
		"{t train_all    |               | train both dictionary and SVM }"
		"{T training_set |               | path to training set root folder, refer to documentation for required content (for training dictionary and/or SVM)}"
		"{I input        |               | path to input image (for inference)}"
		"{G groundtruth  |               | path to input ground truth (for inference evaluation, optional)}"
		"{O output       |               | path to save output image (for inference, optional)}"
		"{S segmentation |               | path to save output segmentation (for inference, optional)}"
		"{show_segments  |false          | show segments after inference? (for inference) }"
		"{show_seg_steps |false          | show all segmentation steps? (for inference) }"
		"{dictionary     |dictionary.yml | dictionary path (default: dictionary.yml) }"
		"{svm            |svm.yml        | SVM path (default: svm.yml) }"
		;
	cv::CommandLineParser parser(argc, argv, keys);

	//only help?
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	int mode;

	cv::String input_path = "", gt_path = "", output_path = "";
	cv::String dictionary_path = "", svm_path = "", trainingset_path = "";
	bool show_segments = false, show_steps = false;

	//inference?
	if (parser.has("i")) {
		mode = MODE::DETECT;

		//we need input path
		if (!parser.has("I")) {
			std::cout << "Input path (-I) not provided." << std::endl;
			return 1;
		}

		input_path = parser.get<cv::String>("I");

		//all else is optional.
		gt_path = parser.get<cv::String>("G"); //"" if not
		output_path = parser.get<cv::String>("O"); //"" if not
		dictionary_path = parser.get<cv::String>("dictionary"); //"dictionary.yml" if not
		svm_path = parser.get<cv::String>("svm"); //"svm.yml" if not
		show_segments = parser.get<bool>("show_segments");
		show_steps = parser.get<bool>("show_seg_steps");
	}
	else if (parser.has("d")) {
		mode = MODE::BUILD_DICTIONARY;

		//we need TS path
		if (!parser.has("T")) {
			std::cout << "Training set path (-T) not provided." << std::endl;
			return 1;
		}

		trainingset_path = parser.get<cv::String>("T");

		//all else is optional.
		dictionary_path = parser.get<cv::String>("dictionary"); //"dictionary.yml" if not
	}
	else if (parser.has("s")) {
		mode = MODE::BUILD_SVM;

		//we need TS path
		if (!parser.has("T")) {
			std::cout << "Training set path (-T) not provided." << std::endl;
			return 1;
		}
		trainingset_path = parser.get<cv::String>("T");

		dictionary_path = parser.get<cv::String>("dictionary"); //"dictionary.yml" if not
		svm_path = parser.get<cv::String>("svm"); //"svm.yml" if not
	}
	else if (parser.has("t")) {
		mode = MODE::BUILD_ALL;

		//we need TS path
		if (!parser.has("T")) {
			std::cout << "Training set path (-T) not provided." << std::endl;
			return 1;
		}
		trainingset_path = parser.get<cv::String>("T");

		dictionary_path = parser.get<cv::String>("dictionary"); //"dictionary.yml" if not
		svm_path = parser.get<cv::String>("svm"); //"svm.yml" if not
	}

	switch (mode) {
	case MODE::DETECT: {
		Detector c;

		// load the input
		cv::Mat src;
		try {
			src = cv::imread(input_path);
			std::cout << "Input loaded. Preprocessing..." << std::endl << std::endl;
		}
		catch (cv::Exception e) {
			std::cout << "Exception during loading: " << e.what() << std::endl;
			return -1;
		}

		// preprocessing: bilateral filtering, meanshift
		cv::Mat processed;
		try {
			cv::Mat src_filt, src_ms;

			std::cout << "Preprocessing: Applying BF" << std::endl;
			Segment::preprocess_BF(src, src_filt);

			std::cout << "Preprocessing: Applying MS" << std::endl;
			Segment::preprocess_MS(src_filt, src_ms);

			std::cout << "Preprocessing: Applying LA" << std::endl;
			Segment::preprocess_LA(src_ms, processed);
		}
		catch (cv::Exception e) {
			std::cout << "Exception during preprocessing: " << e.what() << std::endl;
			return -1;
		}

		// segment extraction
		cv::Mat segments;
		try {
			std::cout << "Preprocessing over. Segmentation..." << std::endl << std::endl;
			Segment::segment(processed, segments, show_steps);
		}
		catch (cv::Exception e) {
			std::cout << "Exception during segmentation: " << e.what() << std::endl;
			return -1;
		}

		// perform inference on segments
		std::vector<bool> results;
		try {
			std::cout << "Segmentation over. Inference..." << std::endl << std::endl;

			for (int i = 0; i < Segment::segments.size(); i++) {
				c.loadDictionary();
				c.loadSVM();

				results.push_back(c.detect(src, Segment::segments.at(i)));
			}
		}
		catch (cv::Exception e) {
			std::cout << "OpenCV exception during inference: " << e.what() << std::endl;
			return -1;
		}
		catch (std::exception e) {
			std::cout << "C++ exception during inference: " << e.what() << std::endl;
			return -1;
		}

		// draw results and also build boat mask, in case it's needed for evaluation
		cv::Mat boatmask = cv::Mat::zeros(src.size(), CV_8U);
		cv::Mat dst;
		try {
			std::cout << "Inference over. Drawing results..." << std::endl << std::endl;

			src.copyTo(dst);
			for (int i = 0; i < results.size(); i++) {
				if (results.at(i)) {
					std::cout << i << ": boat" << std::endl;
					std::vector<std::vector<cv::Point> > contours;
					cv::findContours(Segment::segments.at(i), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

					int largest_area = -1, largest_contour = -1;
					for (int j = 0; j < contours.size(); j++) {
						double a = cv::contourArea(contours.at(j));
						if (a > largest_area) {
							largest_area = a;
							largest_contour = j;
						}
					}

					if (largest_contour == -1) {
						std::cout << "Warning: Cannot draw a rectangle for presumed boat in segment " << i << std::endl;
					}
					else {
						cv::Scalar color(cv::theRNG().uniform(0, 1) * 255, cv::theRNG().uniform(0, 1) * 255, cv::theRNG().uniform(0, 1) * 255); //random color
						cv::Rect r = cv::boundingRect(contours.at(largest_contour));
						cv::rectangle(dst, r, cv::Scalar(0, 255, 0));
						cv::rectangle(boatmask, r, cv::Scalar(255), cv::FILLED);
					}
				}
				else {
					std::cout << i << ": nothing" << std::endl;
				}
			}
		}
		catch (cv::Exception e) {
			std::cout << "OpenCV exception during result drawing: " << e.what() << std::endl;
		}
		catch (std::exception e) {
			std::cout << "C++ exception during result drawing: " << e.what() << std::endl;
		}


		std::cout << "Done!" << std::endl << std::endl;

		// evaluate results if ground truth file loaded
		if (gt_path != "") {
			try {
				double iou = c.evaluate(boatmask, gt_path);
				std::cout << "IoU of detection: " << iou << std::endl;
			}
			catch (cv::Exception e) {
				std::cout << "OpenCV exception during result evaluation: " << e.what() << std::endl;
				return -1;
			}
			catch (std::exception e) {
				std::cout << "C++ exception during result evaluation: " << e.what() << std::endl;
				return -1;
			}
		}

		//write to disk?
		if (output_path != "") {
			try {
				cv::imwrite(output_path, dst);
			}
			catch (cv::Exception e) {
				std::cout << "OpenCV exception during output save: " << e.what() << std::endl;
				return -1;
			}
			catch (std::exception e) {
				std::cout << "C++ exception during output save: " << e.what() << std::endl;
				return -1;
			}
		}

		// Visualize the final image and segmentation, if requested
		if (show_segments) cv::imshow("Segmented", segments);
		cv::imshow("Result", dst);

		cv::waitKey();
		break;
	}
	case MODE::BUILD_DICTIONARY: {
		Trainer c;
		//"B:\\Coding\\computer-vision\\project-boat-detection\\dataset\\TRAINING_DATASET"
		c.loadDatasetPaths(trainingset_path);
		c.trainDictionary(dictionary_path);
		break;
	}
	case MODE::BUILD_SVM: {
		Trainer c;
		c.loadDatasetPaths(trainingset_path);
		c.loadDictionary(dictionary_path);
		c.trainSVM(svm_path);
		break;
	}
	case MODE::BUILD_ALL: {
		Trainer c;
		c.loadDatasetPaths(trainingset_path);
		c.trainDictionary(dictionary_path);
		c.trainSVM(svm_path);
		break;
	}
	}
	return 0;
}