//OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

//STL
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace dnn;

int main(int argc, char* argv[])
{
	//Load Model
	Net net = dnn::readNet("model_matlab.onnx");
	Mat blob = dnn::blobFromImage(cv::imread("image.jpg"), 1.0, Size(224, 224), true, false);

	//Set Input
	net.setInput(blob);

	//Compute Output
	Mat out = net.forward(); //Tensor 1x2x224x224

	//Print results
	ofstream file("results.txt");

	//Print result mask1
	for (int i = 0; i < 224; i++)
	{
		for (int j = 0; j < 224; j++)
		{
			int index[4] = { 0,0,i,j };
			file << "R : " << i << " C : " << j << " value : " << out.at<float>(index) << endl;
		}
	}
	file.close();

	/*
		DO NOT WORK ALSO WITH THIS :
		InputArray arr(out);
		vector<Mat> planes;
		arr.getMatVector(planes);
		InputArray arr1(planes.at(0));
		vector<Mat> planes1;
		arr1.getMatVector(planes1);
	*/
}