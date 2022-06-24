#include "../include/Detector.h"

void Detector::readImages(cv::String pathImages)
{
	//Read all the paths of the images
	std::vector<cv::String> pathSingleImages;
	cv::glob(pathImages, pathSingleImages);

	//Store the images inside the images vector
	for (int i = 0; i < pathSingleImages.size(); i++)
		images.push_back(cv::imread(pathSingleImages.at(i)));
}

std::vector<cv::Mat> Detector::detectHands()
{
	for (int i = 0; i < images.size(); i++)
	{
		getBoudingBoxesDetections(images.at(i));


	}


}

std::vector<cv::Range> Detector::getBoudingBoxesDetections(cv::Mat image)
{
	//Create pyramid
	std::vector<cv::Mat> pyramid = getGaussianPyramid(image);

	//Get for each image in the pyramid the bounding boxes of the hands

	std::tuple<int, int> dimensions = std::make_tuple(image.rows, image.cols);

	for (int i = 0; i < pyramid.size(); i++)
	{
		std::vector<cv::Rect> boundingBoxesHands = getHandsBoundingBoxes(pyramid.at(i),dimensions);
	}

	

	// SLIDING WINDOW
	// DETECTION FOR EACH WINDOW
	// NON MAXIMA SUPPRESSION

	return std::vector<cv::Range>();
}

std::vector<cv::Mat> Detector::getGaussianPyramid(cv::Mat image)
{
	std::vector<cv::Mat> pyramid;
	
	//Add the image as it is
	pyramid.push_back(image);

	cv::Mat temp = image.clone();
	while (true)
	{
		//Compute new rows and cols
		int rows = image.rows / SCALE_PYRAMID;
		int cols = image.cols / SCALE_PYRAMID;

		//Resize the image
		cv::Mat resized;
		cv::resize(temp, resized, cv::Size(cols,rows),cv::INTER_CUBIC);

		//Apply Gaussian Smoothing
		cv::Mat blurred;
		cv::filter2D(resized, blurred, resized.depth(), KERNEL_PYRAMID);

		//Check if the size of the window used for sliding window approach is contained into the image produced
		if (blurred.cols < std::get<0>(WINDOW_SIZE) || blurred.rows < std::get<1>(WINDOW_SIZE))
			break;

		//Add image to the pyramid of images
		pyramid.push_back(blurred);
	}
	return pyramid;
}

std::vector<cv::Rect> Detector::getHandsBoundingBoxes(cv::Mat image, std::tuple<int, int> orginalDimensions)
{
	std::vector<cv::Rect> boundingBoxesHands;

	for (int row = 0; row < image.rows - std::get<0>(WINDOW_SIZE); row += STRIDE_ROWS)
	{
		//Range of rows coordinates
		cv::Range rowRange(row, row + std::get<0>(WINDOW_SIZE));
		
		for (int col = 0; col < image.cols - std::get<1>(WINDOW_SIZE); col += STRIDE_COLS)
		{			
			//Range of cols coordinates
			cv::Range colRange(row, row + std::get<0>(WINDOW_SIZE));

			//Get ROI
			cv::Mat roi = image(rowRange, colRange);

			//Prepare for input to the CNN
			cv::Mat inputCNN = prepareImageForCNN(roi);

			//Get if what it is
			if (isHand(inputCNN))
			{							
				//Need to convert bounding box coordinates to original image size
				
				
				cv::Rect(row,col, std::get<0>(WINDOW_SIZE), std::get<1>(WINDOW_SIZE))
			}
		}
	}



	return std::vector<cv::Range>();
}

cv::Mat Detector::prepareImageForCNN(cv::Mat image)
{
	cv::Mat resized,outputImage;

	//Resize
	cv::resize(image, resized, cv::Size(WIDTH_INPUT_CNN, HEIGHT_INPUT_CNN), cv::INTER_CUBIC);

	//Convert to CV_32
	resized.convertTo(outputImage, CV_32FC3);

	return outputImage;
}

//TODO
bool Detector::isHand(cv::Mat image)
{
	//Read Model
	cv::dnn::Net network = cv::dnn::readNetFromTensorflow(pathModel);
	
	//Set input
	network.setInput(cv::dnn::blobFromImage(image,1.0 / 255.0, cv::Size(WIDTH_INPUT_CNN, HEIGHT_INPUT_CNN), true, false));

	//Forward
	cv::Mat output = network.forward();

	//Here to check??

	return false;
}
