#include "../include/Detector.h"
#include "../include/Utils.h"
#include <iostream>

/**
* This file represent the Detector module "implementation"
* @author : Francesco Caldivezzi
*/

void Detector::readImages(cv::String pathImages)
{
	//Read all the paths of the images
	std::vector<cv::String> pathSingleImages;
	cv::glob(pathImages, pathSingleImages);
	//Store the images inside the images vector
	for (int i = 0; i < pathSingleImages.size(); i++)
	{
		//Split the string
		std::vector<cv::String> splits = Utils::split(pathSingleImages.at(i), '\\');
		images.insert(std::pair<cv::String, cv::Mat>(splits.at(splits.size()-1), cv::imread(pathSingleImages.at(i))));
	}
		
}

std::vector<cv::Mat> Detector::detectHands(cv::String nameImage)
{
	cv::Mat image = images.at(nameImage);
	getBoudingBoxesDetections(image);
}

void Detector::setModel(cv::String pathModel)
{
	this->pathModel = pathModel;
}

std::vector<cv::Range> Detector::getBoudingBoxesDetections(cv::Mat image)
{
	//TODO : WINDOW SIZE DYNAMIC

	//Create pyramid
	std::vector<cv::Mat> pyramid = getGaussianPyramid(image);

	//Get for each image in the pyramid the bounding boxes of the hands
	std::tuple<int, int> dimensions = std::make_tuple(image.rows, image.cols);

	std::vector<cv::Rect> allBoundingBoxesHands;
	std::vector<float> allProbabilities;
	for (int i = 0; i < pyramid.size(); i++)
	{
		std::cout << "COMPUTING BOUNDING BOXES PYRAMID " << i << " OF " << pyramid.size() << std::endl;
		std::vector<float> probabilities;
		std::vector<cv::Rect> boundingBoxes = getHandsBoundingBoxes(pyramid.at(i), dimensions, i,probabilities);
		allBoundingBoxesHands.insert(allBoundingBoxesHands.end(), boundingBoxes.begin(), boundingBoxes.end());
		allProbabilities.insert(allProbabilities.end(), probabilities.begin(), probabilities.end());
	}
	
	//Non maxima Suppression
	std::vector<cv::Rect> finalBoxes = nonMaximaSuppression(allBoundingBoxesHands,allProbabilities);

	
		
	//TODO: REMOVE
	for (int i = 0; i < allBoundingBoxesHands.size(); i++)
	{
		cv::rectangle(image, allBoundingBoxesHands.at(i), cv::Scalar(255, 0, 0));
	}

	cv::imshow("RECTANGLES", image);
	cv::waitKey();


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
		int rows = temp.rows / SCALE_PYRAMID;
		int cols = temp.cols / SCALE_PYRAMID;

		//Resize the image
		cv::Mat resized;
		cv::resize(temp, resized, cv::Size(cols,rows),cv::INTER_CUBIC);

		//Apply Gaussian Smoothing
		cv::Mat blurred;
		cv::filter2D(resized, blurred, resized.depth(), KERNEL_PYRAMID);

		//Check if the size of the window used for sliding window approach is contained into the image produced
		if (blurred.cols < std::get<0>(INITIAL_WINDOW_SIZE) || blurred.rows < std::get<1>(INITIAL_WINDOW_SIZE))
			break;

		//Add image to the pyramid of images
		pyramid.push_back(blurred);

		temp = blurred.clone();
	}
	return pyramid;
}

std::vector<cv::Rect> Detector::getHandsBoundingBoxes(cv::Mat image, std::tuple<int, int> orginalDimensions,int positionPyramid,std::vector<float>& probabilities)
{
	std::vector<cv::Rect> boundingBoxesHands;

	//Compute windows sizes for the current image in the pyramid according to the scale
	int windowSizeHeigth = std::get<0>(INITIAL_WINDOW_SIZE) / std::pow(SCALE_PYRAMID, positionPyramid);
	int windowSizeWidth = std::get<1>(INITIAL_WINDOW_SIZE) / std::pow(SCALE_PYRAMID, positionPyramid);

	//Compute the Stride for the Rows and Cols
	int strideRows = windowSizeHeigth * STRIDE_ROWS_FACTOR;
	int strideCols = windowSizeWidth * STRIDE_ROWS_FACTOR;

	for (int row = 0; row < image.rows - windowSizeHeigth; row += strideRows)
	{
		//Range of rows coordinates
		cv::Range rowRange(row, row + windowSizeHeigth);
		
		for (int col = 0; col < image.cols - windowSizeWidth; col += strideCols)
		{			
			//Range of cols coordinates
			cv::Range colRange(col, col + windowSizeWidth);

			//Get ROI
			cv::Mat roi = image(rowRange, colRange);

			//Prepare for input to the CNN
			cv::Mat inputCNN = prepareImageForCNN(roi);

			//Get output CNN
			std::tuple<bool, float> outputCNN = isHand(inputCNN);
				
			//Get if what it is
			if (std::get<0>(outputCNN))
			{
				//Need to convert bounding box coordinates to original image size
				//(x1,y1) 
				std::tuple<int, int> x1y1 = convertCoordinates(std::tuple<int, int>(col, row),
					orginalDimensions,
					std::tuple<int, int>(image.rows, image.cols));

				//(x2,y2)
				std::tuple<int, int> x2y2 = convertCoordinates(std::tuple<int, int>(col + windowSizeWidth, row + windowSizeHeigth),
					orginalDimensions,
					std::tuple<int, int>(image.rows, image.cols));
				

				//Add Bounding Boxes
				boundingBoxesHands.push_back(cv::Rect(cv::Point(std::get<0>(x1y1), std::get<1>(x1y1)), 
													cv::Point(std::get<0>(x2y2), std::get<1>(x2y2))));

				//Add Probability
				probabilities.push_back(std::get<1>(outputCNN));

				std::cout << "X1 " << std::get<0>(x1y1) << " Y1 " << std::get<1>(x1y1)
					<< " X2 " << std::get<0>(x2y2) << " Y2 " << std::get<1>(x2y2) << std::endl;
			}
		}
	}
	return boundingBoxesHands;
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

std::tuple<bool,float> Detector::isHand(cv::Mat image)
{
	//Read Model
	cv::dnn::Net network = cv::dnn::readNetFromTensorflow(pathModel);
	
	//Set input
	network.setInput(cv::dnn::blobFromImage(image,1.0 / 255.0, cv::Size(WIDTH_INPUT_CNN, HEIGHT_INPUT_CNN), true, false));

	//Forward
	cv::Mat output = network.forward();	

	//Check if it is an hand
	if (output.at<float>(0, 0) > THRESHOLD_DETECTION)
		return std::tuple<bool,float>(false, output.at<float>(0, 0));
	else
		return std::tuple<bool, float>(true, output.at<float>(0, 0));
}

std::tuple<int, int> Detector::convertCoordinates(std::tuple<int, int> coordinatesToConvert, std::tuple<int, int> orginalDimensions, std::tuple<int, int> currentDimensions)
{
	//Convert x coordinate
	int newX = (std::get<0>(coordinatesToConvert) * std::get<1>(orginalDimensions)) / (std::get<1>(currentDimensions));
	if (newX > std::get<1>(orginalDimensions))
		newX = std::get<1>(orginalDimensions);

	//Convert y coordinate
	int newY = (std::get<1>(coordinatesToConvert) * std::get<0>(orginalDimensions)) / (std::get<0>(currentDimensions));
	if (newY > std::get<0>(orginalDimensions))
		newY = std::get<0>(orginalDimensions);

	return std::tuple<int, int>(newX, newY);
}

std::vector<cv::Rect> Detector::nonMaximaSuppression(std::vector<cv::Rect> boxes, std::vector<float> probabilities)
{
	std::vector<cv::Rect> nms;
	//If empty return empty nms
	if (boxes.empty())
		return nms;

	//Convert Bounding Boxes to float coordinates
	std::vector<cv::Rect2f> boxesFloat = convertBoxesToFloatCoordinates(boxes);	
	
	//Get x1s,x2s,y1s,y2s of bounding boxes
	std::vector<float> x1, x2, y1, y2, area, idxsFloat;
	for (int i = 0; i < boxesFloat.size(); i++)
	{
		x1.push_back(boxesFloat.at(i).x);
		y1.push_back(boxesFloat.at(i).y);
		x2.push_back(boxesFloat.at(i).x + boxesFloat.at(i).width);
		y2.push_back(boxesFloat.at(i).y + boxesFloat.at(i).height);
	}

	//Compute area of the bounding boxes
	for (int i = 0; i < boxesFloat.size(); i++)
		area.push_back((x2.at(i) - x1.at(i) + 1) * (x2.at(i) - x1.at(i) + 1));
		
	//If probabilities are present, then use them as idx
	if (!probabilities.empty())	
		idxsFloat.insert(idxsFloat.end(), probabilities.begin(), probabilities.end());
	else //Otherwise use y2 as idx
		idxsFloat.insert(idxsFloat.end(), y2.begin(), y2.end());
	
	std::vector<int> idxs, pickedIndices;

	//Sort the indixes
	idxs = Utils::argSort(idxsFloat);

	while (!idxs.empty())
	{
		//Grab last index in the indexes list and add the index value to the list of picked indices
		int last = idxs.size() - 1;
		int i = idxs.at(last);
		pickedIndices.push_back(i);

		//Slice the idx vector
		std::vector<int> idxsSliced = Utils::slice(idxs, 0, last);		

		//Slice the vectors
		std::vector<float> x1Sliced = Utils::slice(x1, idxsSliced);
		std::vector<float> x2Sliced = Utils::slice(x2, idxsSliced);
		std::vector<float> y1Sliced = Utils::slice(y1, idxsSliced);
		std::vector<float> y2Sliced = Utils::slice(y2, idxsSliced);

		//Find the largest coordinates for the start of the bounding box and the smallest (x,y) coordinates for the end of the bounding box
		std::vector<float> xx1 = Utils::elementWiseMaximum(x1Sliced,x1.at(i));
		std::vector<float> xx2 = Utils::elementWiseMaximum(x2Sliced, x2.at(i));
		std::vector<float> yy1 = Utils::elementWiseMaximum(y1Sliced, y1.at(i));
		std::vector<float> yy2 = Utils::elementWiseMaximum(y2Sliced, y2.at(i));

		//Compute width and heigth of the bounding boxes
		std::vector<float> differenceXX2XX1 = Utils::elementWiseDifference(xx2, xx1);
		std::vector<float> differenceYY2YY1 = Utils::elementWiseDifference(yy2, yy1);
		std::vector<float> w = Utils::elementWiseMaximum(differenceXX2XX1, 0.0f);
		std::vector<float> h = Utils::elementWiseMaximum(differenceYY2YY1, 0.0f);
		
		//Remaining Boxes
		std::vector<cv::Rect> listRemainingBoxes = createListBoxes(xx1, yy1, w, h);
		
		//Selected Rect
		cv::Rect selectedBox(x1.at(i), y1.at(i), x2.at(i) - x1.at(i), y2.at(i) - y1.at(i));

		std::cout << "SELECTED BOX " << selectedBox;

		//Intersection over union
		std::vector<float> ious = intersectionOverUnionElementWise(listRemainingBoxes, selectedBox);
		
		//Update Idx
		std::vector<int> thresholded = Utils::greater(ious, THRESHOLD_OVERLAPPING);
		thresholded.insert(thresholded.begin(), last);
		Utils::deleteElementPositions(idxs, thresholded);
	}

	//Slice boxesFloat and convert to integer coordinates
	std::vector<cv::Rect2f> slicedBoxesFloat = Utils::slice(boxesFloat, pickedIndices);
	return convertBoxesToIntCoordinates(slicedBoxesFloat);	
}

std::vector<cv::Rect2f> Detector::convertBoxesToFloatCoordinates(std::vector<cv::Rect> boxes)
{
	std::vector<cv::Rect2f> boxesFloat;

	for (int i = 0; i < boxes.size(); i++)
		boxesFloat.push_back(cv::Rect2f(boxes.at(i).x, boxes.at(i).y, boxes.at(i).width, boxes.at(i).height));

	return boxesFloat;
}

std::vector<cv::Rect> Detector::convertBoxesToIntCoordinates(std::vector<cv::Rect2f> boxesFloat)
{
	std::vector<cv::Rect> boxes;

	for (int i = 0; i < boxesFloat.size(); i++)
		boxes.push_back(cv::Rect(boxesFloat.at(i).x, boxesFloat.at(i).y, boxesFloat.at(i).width, boxesFloat.at(i).height));

	return boxes;
}

float Detector::intersectionOverUnion(cv::Rect box1, cv::Rect box2)
{
	int xA = std::max(box1.x, box2.x);
	int yA = std::max(box1.y, box2.y);
	int xB = std::min(box1.x + box1.width, box2.x + box2.width);
	int yB = std::min(box1.y + box1.height, box2.y+ box2.height);

	//Area of intersection rectangle
	float intersectionArea = std::max(0, xB - xA) * std::max(0, yB - yA);

	//Area of both boxes
	float boxAreaA = box1.area();
	float boxAreaB = box2.area();

	return intersectionArea / (boxAreaA + boxAreaB - intersectionArea);
}

std::vector<float> Detector::intersectionOverUnionElementWise(std::vector<cv::Rect>& boxes, cv::Rect& box)
{
	std::vector<float> ious;
	
	for (int i = 0; i < boxes.size(); i++)
		ious.push_back(intersectionOverUnion(box, boxes.at(i)));

	return ious;
}

std::vector<cv::Rect> Detector::createListBoxes(std::vector<float>& x1s, std::vector<float>& y1s, std::vector<float>& ws, std::vector<float>& hs)
{
	std::vector<cv::Rect> rectangles;		

	//TODO : ADD CONTROLS SIZE

	for (int i = 0; i < x1s.size(); i++)
		rectangles.push_back(cv::Rect(x1s.at(i), y1s.at(i), ws.at(i), hs.at(i)));

	return rectangles;
}

