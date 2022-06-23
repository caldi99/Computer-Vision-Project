#include "../include/Segment.h"

std::vector<cv::Mat> Segment::segments;

void Segment::preprocess_MS(cv::Mat src, cv::Mat& output, int variant) {

	switch (variant) {
	case 0:
		cv::pyrMeanShiftFiltering(src, output, 40, 25, 3);
		break;
	case 1:
		cv::pyrMeanShiftFiltering(src, output, 20, 30, 3);
		break;
	case 2:
		cv::pyrMeanShiftFiltering(src, output, 20, 20, 3);
		break;
	case 3:
		cv::pyrMeanShiftFiltering(src, output, 20, 10, 3);
		break;
	case 4:
		cv::pyrMeanShiftFiltering(src, output, 10, 30, 3);
		break;
	case 5:
		cv::pyrMeanShiftFiltering(src, output, 10, 20, 3);
		break;
	case 6:
		cv::pyrMeanShiftFiltering(src, output, 10, 10, 3);
		break;
	}
}

void Segment::preprocess_BF(cv::Mat src, cv::Mat& output) {
	cv::bilateralFilter(src, output, 9, 200, 100);
}

void Segment::preprocess_LA(cv::Mat src, cv::Mat& output) {
	//laplacian kernel
	cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1);

	cv::Mat imgLaplacian, src2;
	filter2D(src, imgLaplacian, CV_32F, kernel);
	src.convertTo(src2, CV_32F);
	cv::Mat sharpened = src2 - imgLaplacian;

	sharpened.convertTo(output, CV_8UC3);
}

void Segment::segment(cv::Mat src, cv::Mat& segment_map, bool show_steps) {

	cv::Size img_size = src.size(); //one-point stop for every time size has to be referenced... (just for clarity)

	//get normalized grayscale version
	cv::Mat bw;
	cvtColor(src, bw, cv::COLOR_BGR2GRAY);
	cv::normalize(bw, bw, 0, 255, cv::NORM_MINMAX);

	//show segmentation input
	if (show_steps) {
		cv::imshow("Segmentation input", bw);
		cv::waitKey();
	}

	// FLOODFILL CENTROID GENERATION
	std::cout << "Segmentation: floodfill centroid generation" << std::endl;

	int levels = 16;
	int step = 256 / levels;
	std::vector<cv::Point> centroids; //centroids of large contiguous areas will be seed points for segmentation
	cv::Mat mask_result = cv::Mat::zeros(img_size, CV_32S); //in case we want to visualize all contours

	for (int i = 0; i < levels; i++) {
		//get parts of image at this quantized level (level mask)
		cv::Mat mask;
		cv::inRange(bw, i * step, (i + 1) * step - 1, mask);

		//display level mask
		if (show_steps) {
			cv::imshow("Level mask", mask);
			cv::waitKey();
		}

		//find contours in this level
		std::vector<std::vector<cv::Point> > contours;
		cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		cv::Mat contourmap = cv::Mat::zeros(img_size, CV_32S);

		//for each contour in this level, extract a centroid
		for (size_t j = 0; j < contours.size(); j++) {
			//first, discard small contours
			if (cv::contourArea(contours.at(j)) < 250) continue;

			//in case we want to visualize the contours
			if (show_steps)
				drawContours(contourmap, contours, static_cast<int>(j), cv::Scalar(static_cast<int>(j) + 1), -1);

			//calculate centroids
			if (contours.at(j).size() > 2) { //sanity check
				cv::Moments m = cv::moments(contours.at(j));
				int cX = m.m10 / m.m00;
				int cY = m.m01 / m.m00;
				centroids.push_back(cv::Point(cX, cY));
			}
		}

		//visualize contours at this level
		if (show_steps) {
			cv::Mat contourmap_toshow;
			contourmap.convertTo(contourmap_toshow, CV_8U, 10.0);
			cv::imshow("Contours", contourmap_toshow);
			cv::waitKey();
		}

		//aggregate contours for visualization
		if (show_steps) cv::bitwise_or(mask_result, contourmap, mask_result);
	}

	//visualize all kept contours at all levels (their centroids will be used for floodfill)
	if (show_steps) {
		cv::Mat resultmap_toshow;
		mask_result.convertTo(resultmap_toshow, CV_8U, 10.0);
		cv::imshow("Result", resultmap_toshow);
		cv::waitKey();
	}


	// LARGE SEGMENT EXTRACTION (FLOODFILL)
	std::cout << "Segmentation: large segment extraction via floodfill" << std::endl;
	std::vector<cv::Mat> masks; //floodfill masks: these will eventually define segments
	for (int i = 0; i < centroids.size(); i++) {
		//prepare floodfill mask for this centroid (requires 2 extra r/c)
		cv::Mat ff_mask = cv::Mat::zeros(bw.rows + 2, bw.cols + 2, CV_8U);

		//flags: 8: compare diagonals | mask_only: fill the mask instead of the og image | 255: color for mask
		cv::floodFill(bw, ff_mask, centroids.at(i), cv::Scalar(255), 0, cv::Scalar(5, 5, 5), cv::Scalar(5, 5, 5), 8 | cv::FLOODFILL_MASK_ONLY | (255 << 8));

		//remove extra rows/cols added to make floodfill work
		//first problem: since 1px border nonzero, this screws with similarity computation
		//second problem (later): a mask must be the correct size for image to be successfully manipulated (and also we don't want an extraneous outside border in bounding box computation)
		cv::Mat cropped_mask = ff_mask(cv::Range(1, ff_mask.rows - 1), cv::Range(1, ff_mask.cols - 1)).clone();

		//filter out very small masks (less than S.P.T. area)
		float smallness_threshold = 0.01;
		if (cv::countNonZero(cropped_mask) < cropped_mask.rows * cropped_mask.cols * smallness_threshold) continue;

		//at this point, FF mask = cluster/segment mask
		//visualize this cluster
		if (show_steps) {
			cv::imshow("FF mask", cropped_mask);
			cv::waitKey();
		}

		//finally, 
		masks.push_back(cropped_mask);
	}

	//discard duplicate/"too similar" masks according to some similarity metric (many floodfilled areas will be the same if centroids belong to the same one)
	//idea: get area of mask A and B, then compute A & B (=intersection), merge/delete if percentage area of A & B above of threshold

	std::cout << "Initially found " << masks.size() << " large segments." << std::endl;

	int deleted_masks = 0; //purely informational
	for (int i = 0; i < masks.size(); i++) {
		cv::Mat A = masks.at(i);
		std::vector<int> similar_masks;
		//on each pass of the mask set, gather all masks similar to mask(i), then delete them at the end of the pass
		//note that we start from j = i+1 as to avoid comparing two masks twice and also issues with deletions
		for (int j = i + 1; j < masks.size(); j++) {
			cv::Mat B = masks.at(j);
			cv::Mat I;
			cv::bitwise_and(A, B, I);

			int area_a = cv::countNonZero(A);
			int area_b = cv::countNonZero(B);
			int area_i = cv::countNonZero(I);

			if (area_i == 0) continue; //don't even bother computing division

			//it seems that floodfill always results in partitions, so either area_i = 0, or A = B and then area_i = area_a = area_b.
			//this probably depends on floodfill flags (there's an option to compare every point with the centroids rather than from nearest neighbour)
			//(if that was the case we used, this wouldn't be true anymore. and also exact centroid position would affect result much more!)
			//cannot guarantee this is true, so to be safe, still compute a metric
			//good idea also in case the other option is ever used?

			//area_i is upperbounded by lesser of area_a, area_b (when B entirely in A or v.v.)
			double metric = (double)area_i / (double)(area_a < area_b ? area_a : area_b);

			//std::cout << "found nonzero intersection masks: " << i << " and " << j << std::endl;
			//std::cout << area_a << " - " << area_b << " - " << area_i << "(" << metric << ")" << std::endl;
			//cv::imshow("A", A);
			//cv::imshow("B", B);
			//cv::imshow("I", I);
			//cv::waitKey();

			if (metric > 0.5)
				similar_masks.push_back(j);
		}

		//delete duplicates to mask(i)
		for (int j = 0; j < similar_masks.size(); j++)
			// don't forget to take into account that, in each iteration, your target has moved j steps to the left because of previous deletions
			// solution: either delete in reverse order (higher elements first) or subtract count of deleted items (-j) from index
			masks.erase(masks.begin() + similar_masks.at(j) - j);

		//update counter (informational only)
		deleted_masks += similar_masks.size();
	}

	std::cout << "Deleted " << deleted_masks << " duplicate segment(s)." << std::endl;

	//display what we have at this point (large clusters without the duplicates)
	if (show_steps) {
		for (int i = 0; i < masks.size(); i++) {
			cv::imshow("Cluster", masks.at(i));
			cv::waitKey();
		}
	}


	// CONTIGUOUS SMALL SEGMENT EXTRACTION
	//now, we ignored masks with area less than a threshold
	//suppose we have a very uneven object which is actually pretty well distinct from background. (boat or boat parts!)
	//so ideally this is surrounded by large clusters (eg. the sea and the sky)
	//idea:
	// 	   get all big cluster masks (ideally including the sky, the sea, the big parts of the boat(s), maybe the trees, etc.)
	// 	   sum them (bitwise or) and get the negative (bitwise not)
	// 	   this will result in a "mask of small masks"
	// 	   extract (large) connected components
	std::cout << "Segmentation: contiguous small segment extraction" << std::endl;

	//building "mask of small masks"
	cv::Mat mask_of_small_masks = cv::Mat::zeros(bw.rows, bw.cols, CV_8U);
	for (int i = 0; i < masks.size(); i++)
		cv::bitwise_or(masks.at(i), mask_of_small_masks, mask_of_small_masks);
	cv::bitwise_not(mask_of_small_masks, mask_of_small_masks);

	//erode to remove loose connections
	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(7, 7));
	erode(mask_of_small_masks, mask_of_small_masks, element);

	//display "mask of small masks"
	if (show_steps) {
		cv::imshow("Small parts", mask_of_small_masks);
		cv::waitKey();
	}

	//extract (large) connected components
	std::vector<std::vector<cv::Point> > small_contours;
	cv::findContours(mask_of_small_masks, small_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat small_contourmap = cv::Mat::zeros(img_size, CV_32S);
	int new_clusters = 0; //purely informative
	for (size_t i = 0; i < small_contours.size(); i++) {
		//first, discard small contours, again according to some threshold (higher, since these clusters are aggregations of smaller clusters)
		float smallness_threshold = 0.01;
		if (cv::contourArea(small_contours.at(i)) < mask_of_small_masks.rows * mask_of_small_masks.cols * smallness_threshold) continue;

		//did this survive? add it to the big clusters
		new_clusters++;
		cv::Mat new_cluster = cv::Mat::zeros(img_size, CV_32S);
		cv::drawContours(new_cluster, small_contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
		new_cluster.convertTo(new_cluster, CV_8U, 10.0); //needs to be the proper format
		masks.push_back(new_cluster);

		cv::drawContours(small_contourmap, small_contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED); //for visualization
	}

	std::cout << "Found " << new_clusters << " new segment(s) with small segment extraction." << std::endl;
	std::cout << "Final number of segments: " << masks.size() << std::endl;

	//display new clusters
	if (show_steps) {
		cv::Mat contourmap_toshow;
		small_contourmap.convertTo(contourmap_toshow, CV_8U, 10.0);
		cv::imshow("New clusters", contourmap_toshow);
		cv::waitKey();
	}

	//display all (final) clusters
	if (show_steps) {
		for (int i = 0; i < masks.size(); i++) {
			cv::imshow("Cluster", masks.at(i));
			cv::waitKey();
		}
	}

	//create segment map for visualization (secondary output)
	//useful to judge quality at a first glance
	segment_map = cv::Mat::zeros(img_size, CV_8UC3);
	for (int i = 0; i < masks.size(); i++) {
		cv::Scalar color(cv::theRNG().uniform(0, 256), cv::theRNG().uniform(0, 256), cv::theRNG().uniform(0, 256)); //random color

		std::vector<std::vector<cv::Point> > segment_map_contours;
		cv::findContours(masks.at(i), segment_map_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		for (size_t j = 0; j < segment_map_contours.size(); j++)
			cv::drawContours(segment_map, segment_map_contours, static_cast<int>(j), color, cv::FILLED);
	}

	//save results!
	segments = masks;
}
/*
	// Generate random colors
	std::vector<cv::Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++)
	{
		int b = cv::theRNG().uniform(0, 256);
		int g = cv::theRNG().uniform(0, 256);
		int r = cv::theRNG().uniform(0, 256);
		colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
	// Create the result image
	dst = cv::Mat::zeros(markers.size(), CV_8UC3);
	// Fill labeled objects with random colors
	for (int i = 0; i < markers.rows; i++)
	{
		for (int j = 0; j < markers.cols; j++)
		{
			int index = markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(contours.size()))
			{
				dst.at<cv::Vec3b>(i, j) = colors[index - 1];
			}
		}
	}
}*/