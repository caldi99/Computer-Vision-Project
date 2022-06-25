//MYLIB
#include "../include/Detector.h"
#include "../include/Segmentator.h"

//STL
#include <iostream>

//NAMESPACES
using namespace cv;
using namespace std;

int mainDaniela();
int mainFinale(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	//mainDaniela();
	mainFinale(argc, argv);
	
}

int mainDaniela()
{
	Segmentator segmentator;
	segmentator.segment_1("01.jpg");
	cv::waitKey(0);
	return 0;
}

enum MODE {
	DETECT,SEGMENT,ERROR
};

int mainFinale(int argc, char* argv[]) 
{
	//one dash in front if single letters, two dashes if words
	const String KEYS =
		//COMMON
		"{help h usage ?||print this message}"
		"{m model|../model/model.pb| path to the model}"
		"{a annotations|../testset/det/| path to the test set annotations}"
		"{i images|../testset/rgb/| path to the test set images}"
		"{d detect || run detection mode}"	
		"{n name |28| name of the image for which applying detection}" //if only this, show the image with detection only 
		"{opd || path where the image with inside the detections will be stored }" //output path detections ../detections/
		"{opious || path where the ious results for the image will be stored }"		//output path ious ../ious/
		
		//TODO ADD PARAMETERS FOR SEGMENTATIONS
		"{s segment || run segmentation mode}"
		;

	//Parse command line
	CommandLineParser parser(argc, argv, KEYS);

	//If help print infos
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	//Choose Mode
	int mode;
	

	if (parser.has("d"))
		mode = MODE::DETECT;
	else if (parser.has("s"))
		mode = MODE::SEGMENT;
	else
		mode = MODE::ERROR;

	//Declare variables
	Detector detector;


	switch (mode)
	{
		case MODE::SEGMENT:
				//TODO : CODE TO ADD
			break;
		case MODE::DETECT:
		{
			//TODO: INTERSECTION OVER UNION OF TWO SETS (NEED TO ORDER THEM BASED ON (X1,Y1)
			detector.setModel(parser.get<String>("m"));
			detector.readGroundTruth(parser.get<String>("a"));
			detector.readImages(parser.get<String>("i"));
			
			//Get Image Name
			String imageName = parser.get<String>("n");

			//Get Image By Name
			Mat image = detector.getImgeByName(imageName);
			
			//Detect Bounding Boxes Image
			vector<Rect> boundingBoxes = detector.detectHands(imageName);

			if (parser.has("opd") && parser.has("opious")) //Both
			{

			}
			else if (parser.has("opd")) //only
			{

			}
			else if (parser.has("opious"))
			{

			}

			//In each situation show the image and its boxes 
			for (int i = 0; i < boundingBoxes.size(); i++)
				rectangle(image, boundingBoxes.at(i), cv::Scalar(255, 0, 0));
			imshow(imageName, image);
			waitKey();
			break;
		}
		case MODE::ERROR:
		{
			cout << "Error, You need to execute this file by adding into the command line either -detect or -segment or both";
			break;
		}			
	}
	return 0;
}


