//MYLIB
#include "../include/Detector.h"


//NAMESPACES
using namespace cv;

int main(int argc, char* argv[]) 
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



	return 0;
}
