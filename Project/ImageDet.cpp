cv::Mat ImageDet::drawingNegBoundingBox(cv::Mat& image, std::string filename)
{

	//Image that will be returned
	cv::Mat Output = image.clone();

	//Containing bounding box coordinates
	std::vector<int> coordinates;
	int number;

	//Reading (all) the bounding box coordinates, they are saved in the numbers vector
	std::ifstream input_file(filename);
	if (!input_file.is_open()) {
		std::cerr << "Could not open the file - '"
			<< filename << "'" << std::endl;
	}

	while (input_file >> number) {
		coordinates.push_back(number);
	}

	input_file.close();


	/*
		coordinates vector size must be a multiple of 4.

		From vector of coordinates we consider 4 element at times:

		elem at position i --> x coordinates of top left corner
		elem at position i+1 --> y coordinates of top left corned
		elem at position i+2 --> Width of bounding box
		elem at position i+3 --> Height of bounding box
	*/
	if (coordinates.size() % 4 != 0)
		std::cerr << "The file containing bounding box is not well formatted" << std::endl;


	for (int i = 0; i < coordinates.size(); i = i + 4) {


		//Calculating the 4 points of bounding box
		cv::Point A = cv::Point(coordinates[i], coordinates[i + 1]);  //alto sx
		cv::Point B = cv::Point(coordinates[i] + coordinates[i + 2], coordinates[i + 1]); //alto dx
		cv::Point C = cv::Point(coordinates[i] + coordinates[i + 2], coordinates[i + 1] + coordinates[i + 3]); //basso dx
		cv::Point D = cv::Point(coordinates[i], coordinates[i + 1] + coordinates[i + 3]); //basso sx

		//Drawing lines

		for (int i = A.x; i < B.x; i++) {
			for (int j = A.y; j < D.y; j++) {
				Output.at<cv::Vec3b>(i, j)[0] = 0;
				Output.at<cv::Vec3b>(i, j)[1] = 0;
				Output.at<cv::Vec3b>(i, j)[2] = 0;

			}
		}
	}

	return Output;

}