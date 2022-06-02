#include "../include/ImageDetection.h"


cv::Mat ImageDetection::drawingBoundingBox(cv::Mat& image, std::string filename)
{
	
	//Image with bounding box
	cv::Mat imgWithBoundingbox = image.clone();
	int bbCoordinates[4];

	
	//Containing bounding box coordinates
	std::vector<int> numbers;
	int number;

	std::ifstream input_file(filename);
	if (!input_file.is_open()) {
		std::cerr << "Could not open the file - '"
			<< filename << "'" << std::endl;
	}

	while (input_file >> number) {
		numbers.push_back(number);
	}

	input_file.close();

	for (int i = 0; i < numbers.size(); i = i + 4) {


		bbCoordinates[0] = numbers[i];
		bbCoordinates[1] = numbers[i + 1];
		bbCoordinates[2] = numbers[i + 2];
		bbCoordinates[3] = numbers[i + 3];

		//Calculating points
		cv::Point A = cv::Point(bbCoordinates[0], bbCoordinates[1]);
		cv::Point B = cv::Point(bbCoordinates[0] + bbCoordinates[2], bbCoordinates[1]);
		cv::Point C = cv::Point(bbCoordinates[0] + bbCoordinates[2], bbCoordinates[1] + bbCoordinates[3]);
		cv::Point D = cv::Point(bbCoordinates[0], bbCoordinates[1] + bbCoordinates[3]);

		//Drawing lines
		cv::line(imgWithBoundingbox, A, B, COLORBB);
		cv::line(imgWithBoundingbox, B, C, COLORBB);
		cv::line(imgWithBoundingbox, C, D, COLORBB);
		cv::line(imgWithBoundingbox, D, A, COLORBB);

	}

	return imgWithBoundingbox;

}

