#ifndef UTILSHOUGHTRANSFORM_H
#define UTILSHOUGHTRANSFORM_H

#include <opencv2/core/base.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


class UtilsHoughTransform
{
public:
	UtilsHoughTransform();

	void generalizedHoughTransform(cv::Mat pattern, cv::Mat img);

private:

};





#endif UTILSHOUGHTRANSFORM_H
