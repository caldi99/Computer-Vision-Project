#include "../include/ConvexHull.h"

UtilsConvexHull::UtilsConvexHull()
{
}

void UtilsConvexHull::computeHull(cv::Mat source)
{
	cv::Mat cannyImage;
	cv::Canny(source, cannyImage,150, 255);

    cv::imshow("Hull demo", cannyImage);
    cv::waitKey();


    std::vector<std::vector<cv::Point> > contours;

    cv::findContours(cannyImage, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    
	

    std::vector<std::vector<cv::Point>>hull(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {        
        cv::convexHull(contours[i], hull[i]);
    }
    
    cv::Mat drawing = cv::Mat::zeros(cannyImage.size(), CV_8UC3);

    cv::RNG rng(12345);

    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(drawing, contours, (int)i, color);
        drawContours(drawing, hull, (int)i, color);
    }
    cv::imshow("Hull demo", drawing);
    cv::waitKey();

}
