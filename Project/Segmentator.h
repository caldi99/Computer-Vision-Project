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


//MYLIB
#include "Utils.h"

//TODO THINK OF WHAT CAN BE PASSED AS REFERENCE

/**
* This class represent the Detector Module "definitions"
* @author : Francesco Caldivezzi
*/
class Segmentator
{
public:

	/**
	* This function will compute our first idea of segmentator:
	* apply bilateral filter
	* apply find contour
	* thershold
	* @param pathImages : The path were the images are
	*/
	void segment_1(cv::String pathImages);
}