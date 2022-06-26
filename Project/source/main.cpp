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
		"{n name |27| name of the image for which applying detection / segmentation}" 

		//Detection Parameters
		"{d detect || run detection mode}"			
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
		{
			cout << "YOU SELECTED SEGMENTATION MODE " << endl;

			//TODO : CODE TO ADD
			break;
		}			
		case MODE::DETECT: //This part was entirly written by Francesco Caldivezzi
		{
			cout << "YOU SELECTED DETECTOR MODE " << endl;

			detector.setModel(parser.get<String>("m"));
			detector.readGroundTruth(parser.get<String>("a"));
			detector.readImages(parser.get<String>("i"));
			
			//Get Image Name
			String imageName = parser.get<String>("n");

			//Get Image By Name
			Mat image = detector.getImgeByName(imageName);
			
			cout << "DETECTOR IS RUNNING " << endl;

			//Detect Bounding Boxes Image
			vector<Rect> boundingBoxes = detector.detectHands(imageName);

			String outputPathIous, outputPathDetections;

			if (parser.has("opd") && parser.has("opious")) //Both
			{
				outputPathDetections = parser.get<String>("opd");
				outputPathIous = parser.get<String>("opious");
				cout << "SAVING DETECTIONS IN " + outputPathDetections << endl;
				cout << "SAVING IOUS IN " + outputPathIous << endl;
				detector.saveDetections(outputPathDetections, imageName, boundingBoxes);
				detector.saveIntersectionsOverUnions(outputPathIous, imageName, boundingBoxes);
			}
			else if (parser.has("opd")) //only
			{
				outputPathDetections = parser.get<String>("opd");
				cout << "SAVING DETECTIONS IN " + outputPathDetections << endl;
				detector.saveDetections(outputPathDetections, imageName, boundingBoxes);
			}
			else if (parser.has("opious"))
			{				
				outputPathIous = parser.get<String>("opious");
				cout << "SAVING IOUS IN " + outputPathIous << endl;
				detector.saveIntersectionsOverUnions(outputPathIous, imageName, boundingBoxes);
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