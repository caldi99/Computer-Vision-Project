//MYLIB

#include "../include/Detector.h"
#include "../include/Segmentator.h"


//NAMESPACES
using namespace cv;

int mainDaniela();
int mainFrancesco();
int main()
{
	mainDaniela();
	//mainFrancesco();
	
}

int mainDaniela()
{
	Segmentator segmentator;
	segmentator.segment_1("01.jpg");
	cv::waitKey(0);
	return 0;
}
int mainFrancesco()
{

	Detector detector;
	detector.readImages("../testset/rgb/");
	detector.setModel("../model/model.pb");
	detector.detectHands("01");
}
/*
int mainFrancesco(int argc, char* argv[]) 
{
	const String keys =
		"{help h usage ? |				    | print this message}"
		"{detect		 |	         	    | run detection mode}"
		"{segment		 |                  | run segmentation mode}"
		"{iou			 |	                | compute intersection over union for each detected hand}"
		"{pxa			 |	         	    | compute pixel accuracy for each segmented hand}"
		"{@images		 |../testset/rgb/   | path to the test set images}"
		"{@annotations   |../testset/det/   | path to the test set images}"
		"{@model         |../model/model.pb | path to the CNN model}"
		;

	CommandLineParser parser(argc, argv, keys);

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	int x = 0;
	//std::tuple<int, int>(x, x);

	return 0;
}*/
