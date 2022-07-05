#ifndef PREPROCSEGMENTATOR_H
#define PREPROCSEGMENTATOR_H

//OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/core.hpp"
/**
* This class perform preprocessing operations for segmentation task
* @author : Simone D'Antimo
*/
class PreProcSegmentator
{
public:

	/**
	* Returns 2 images , binary and Laplacian filtered one
	*/
	static void getPreFiltered(cv::Mat& src, cv::Mat& binaryDst, cv::Mat& laplacianDst);

};
#endif 
