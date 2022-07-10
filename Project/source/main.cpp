//MYLIB
#include "../include/Detector.h"
#include "../include/Segmentator.h"
#include "../include/PreProcSegmentator.h"

//STL
#include <iostream>

//NAMESPACES
using namespace cv;
using namespace std;

int mainDaniela();
int mainFinale(int argc, char* argv[]);
//void detectAllImages();

int main(int argc, char* argv[])
{
	//mainDaniela();
	mainFinale(argc, argv);
	//detectAllImages();
	
}

int mainDaniela()
{
	Segmentator segmentator;
	segmentator.segment_1("01.jpg");
	/*
	Mat src = imread("01.jpg", IMREAD_COLOR);
	Mat bin;
	Mat lap;
	PreProcSegmentator p;
	p.getPreFiltered(src, bin, lap);
	imshow("b", bin);
	imshow("l", lap);
	*/
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
		"{help h usage ?||print this message }"
		"{m model|| path to the model }"		
		"{a annotation|| path to one of test set annotation or mask depeding if the program is being run in detection or segmentation mode }"
		"{i image|../testset/rgb/01.jpg| path to one of test set image }"

		//Detection Parameters		
		"{d detect || run detection mode}"					
		"{opd || path where the image with inside the detections will be stored }" //output path detections ../detections/
		"{opious || path where the ious results for the image will be stored }"		//output path ious ../ious/
		
		//Segmentation Paramaeters
		"{s segment || run segmentation mode}"
		"{ops || path where the image with inside segmentations will be stored }" // output path segmentations ../segmentations/
		"{oppa || path where pixel accuracy results for the image will be stored }" // output path pixel accuracies ../pixelaccuracies/
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
	Segmentator segmentator;

	try
	{
		switch (mode)
		{
			case MODE::SEGMENT: //This part was entirly written by Francesco Caldivezzi
			{
				cout << "YOU SELECTED SEGMENTATION MODE " << endl;

				segmentator.setModel(parser.get<String>("m"));
				segmentator.readGroundTruth(parser.get<String>("a"));
				segmentator.readImage(parser.get<String>("i"));

				//Get Image 
				Mat image = segmentator.getImage();

				cout << "SEGMENTATOR IS RUNNING " << endl;

				//TODO : think to what functions to call from here on

				if (parser.has("ops") && parser.has("oppa")) 
				{
				}
				else if (parser.has("ops")) //only
				{
				}
				else if (parser.has("oppa"))
				{
				}
				else
				{
					cout << "Error, You need to execute this file by adding into the command line either -ops or -oppa or both";
					return 1;
				}
				//TODO: Show the final image here with hands segmentated

				break;
			}
			case MODE::DETECT: //This part was entirly written by Francesco Caldivezzi
			{
				cout << "YOU SELECTED DETECTOR MODE " << endl;

				detector.setModel(parser.get<String>("m"));
				detector.readGroundTruth(parser.get<String>("a"));
				detector.readImage(parser.get<String>("i"));				

				//Get Image 
				Mat image = detector.getImage();

				cout << "DETECTOR IS RUNNING " << endl;

				//Detect Bounding Boxes Image
				vector<Rect> boundingBoxes = detector.detectHands();

				String outputPathIous, outputPathDetections;

				if (parser.has("opd") && parser.has("opious")) 
				{
					outputPathDetections = parser.get<String>("opd");
					outputPathIous = parser.get<String>("opious");
					cout << "SAVING DETECTIONS IN " + outputPathDetections << endl;
					cout << "SAVING IOUS IN " + outputPathIous << endl;
					detector.saveDetections(outputPathDetections, boundingBoxes);
					detector.saveIntersectionsOverUnions(outputPathIous, boundingBoxes);
				}
				else if (parser.has("opd")) 
				{
					outputPathDetections = parser.get<String>("opd");
					cout << "SAVING DETECTIONS IN " + outputPathDetections << endl;
					detector.saveDetections(outputPathDetections, boundingBoxes);
				}
				else if (parser.has("opious"))
				{
					outputPathIous = parser.get<String>("opious");
					cout << "SAVING IOUS IN " + outputPathIous << endl;
					detector.saveIntersectionsOverUnions(outputPathIous, boundingBoxes);
				}
				else
				{
					cout << "Error, You need to execute this file by adding into the command line either -opd or -opious or both";
					return 1;
				}

				//In each situation show the image and its boxes 
				for (int i = 0; i < boundingBoxes.size(); i++)
					rectangle(image, boundingBoxes.at(i), cv::Scalar(255, 0, 0));
				imshow("Image", image);
				waitKey();
				break;
			}
			case MODE::ERROR:
			{
				cout << "Error, You need to execute this file by adding into the command line either -detect or -segment";
				return 1;
			}
		}
	}
	catch (exception e)
	{
		cout << "AN ERROR OCCURED : " << endl << e.what();
		return 1;
	}	
	return 0;
}


/*void detectAllImages()
{
	//Create Detector
	Detector detector;

	//Set Model
	detector.setModel("../model/model.pb");
	
	for (int i = 1; i <= 30; i++)
	{
		//Create Image Name
		String imageName;
		if (i < 10)
			imageName = "0" + std::to_string(i);
		else		
			imageName = std::to_string(i);
		
		String pathImage = "../testset/rgb/" + imageName + ".jpg";
		String pathGt = "../testset/det/" + imageName + ".txt";

		//Read image and gt
		detector.readImage(pathImage);
		detector.readGroundTruth(pathGt);

		cout << "DETECTOR IS RUNNING FOR : "<< std::to_string(i) << endl;

		//Compute bounding boxes
		vector<Rect> boundingBoxes = detector.detectHands();
		
		String outputPathIous = "../ious/";
		String outputPathDetections = "../detections/";

		//Save results
		detector.saveDetections(outputPathDetections, boundingBoxes);
		detector.saveIntersectionsOverUnions(outputPathIous, boundingBoxes);		
	}
}*/