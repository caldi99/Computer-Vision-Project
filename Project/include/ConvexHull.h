#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <vector>

class UtilsConvexHull
{

public :
	UtilsConvexHull();

	void computeHull(cv::Mat source);

};