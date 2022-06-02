#ifndef BAGOFWORDS_H
#define BAGOFWORDS_H

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

class BagOfWords
{
public:
	BagOfWords(std::string pathDataset);

	void trainBagOfWords();


private:

	std::vector<cv::Mat> trainingSet;
	std::vector<cv::Mat> descriptors;
	std::vector<std::vector<cv::KeyPoint>> keypoints;
};
#endif
