#ifndef SEGMENTATOR_H
#define SEGMENTATOR_H

//OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

//STL
#include <vector>
#include <tuple>
#include <unordered_map>


//TODO THINK OF WHAT CAN BE PASSED AS REFERENCE

/**
* This class represent the Segmentaor Module 
* @author : Daniela Cuza
*/
class Segmentator
{
public:

	/**
	* This function will read the images inside pathImages
	* @param pathImages : The path were the images are
	*/
	void segment_1(cv::String pathImage);
};
#endif // !SEGMENTATOR_H