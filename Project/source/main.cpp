//MYLIB
#include "../include/Detector.h"

//STL
#include <iostream>

//NAMESPACES
using namespace cv;
using namespace std;

int mainDaniela();
int mainFrancesco();
int mainFinale(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	//mainDaniela();
	//mainFrancesco();
	mainFinale(argc,argv);
	
}

int mainDaniela()
{
	//Here my code
	return 0;
}
int mainFrancesco()
{

	Detector detector;
	detector.readImages("../testset/rgb/");
	
	detector.setModel("../model/model.pb");
	detector.detectHands("01");
}

enum MODE {
	DETECT,SEGMENT,BOTH,ERROR
};

int mainFinale(int argc, char* argv[]) 
{
	//one dash in front if single letters, two dashes if words
	const String keys =
		"{help h usage ?||print this message}"
		"{d detect || run detection mode}"
		"{s segment || run segmentation mode}"
		"{m model|../model/model.pb| path to the model}"
		"{a annotations|../testset/det/| path to the test set annotations}"
		"{i images|../testset/rgb/| path to the test set images}"
		"{}"
		/*"{iou			 |	                | compute intersection over union for each detected hand}"
		"{pxa			 |	         	    | compute pixel accuracy for each segmented hand}"
		"{@images		 |../testset/rgb/   | path to the test set images}"
		"{@annotations   |../testset/det/   | path to the test set images}"
		"{@model         |../model/model.pb | path to the CNN model}"*/
		;

	//Parse command line
	CommandLineParser parser(argc, argv, keys);

	//If help print infos
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	//Choose Mode
	int mode;
	
	if (parser.has("d") && parser.has("s"))
		mode = MODE::BOTH;
	else if (parser.has("d"))
		mode = MODE::DETECT;
	else if (parser.has("s"))
		mode = MODE::SEGMENT;
	else
		mode = MODE::ERROR;

	//Declare variables
	Detector detector;

	switch (mode)
	{
		case MODE::BOTH :
				//TODO : CODE TO ADD
			break;
		case MODE::SEGMENT:
				//TODO : CODE TO ADD
			break;
		case MODE::DETECT:
				//TODO: INTERSECTION OVER UNION OF TWO SETS (NEED TO ORDER THEM BASED ON (X1,Y1)
				detector.setModel(parser.get<String>("m"));
				detector.readGroundTruth(parser.get<String>("a"));
				detector.readImages(parser.get<String>("i"));

			break;
		case MODE::ERROR:
			cout << "Error, You need to execute this file by adding into the command line either -detect or -segment or both";
			return 0;			
	}
	return 0;
}
