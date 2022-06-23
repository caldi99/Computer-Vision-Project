#include "../include/Detector.h"

Detector::Detector() {

}

bool Detector::loadDictionary(std::string dictionary_path) {
    try {
        cv::FileStorage fs(dictionary_path, cv::FileStorage::READ);
        fs["dictionary"] >> dictionary;
        fs.release();
        return true;
    }
    catch (cv::Exception e) {
        std::cout << "Error loading dictionary: " << e.what() << std::endl;
        return false;
    }
}
bool Detector::loadSVM(std::string svm_path) {
    try {
        svm = cv::ml::SVM::create();
        svm = cv::Algorithm::load<cv::ml::SVM>(svm_path);

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED); //bf should be fine? otherwise flann
        extractor = cv::SIFT::create();
        bow_extractor = new cv::BOWImgDescriptorExtractor(extractor, matcher);
        bow_extractor->setVocabulary(dictionary);
        return true;
    }
    catch (cv::Exception e) {
        std::cout << "Error loading SVM: " << e.what() << std::endl;
        return false;
    }
}

//inference (no mask)
bool Detector::detect(cv::Mat input) {
    cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoint;
    cv::Mat descriptor;
    descriptor.convertTo(descriptor, CV_32F);
    sift->detectAndCompute(input, cv::Mat(), keypoint, descriptor); //descriptor is gonna be overwritten
    bow_extractor->compute(input, keypoint, descriptor);

    if (descriptor.rows > 0 || descriptor.cols > 0) return svm->predict(descriptor);
    else return false;
}
//inference (mask)
bool Detector::detect(cv::Mat input, cv::Mat mask) {
    cv::Ptr<cv::Feature2D> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoint;
    cv::Mat descriptor;
    descriptor.convertTo(descriptor, CV_32F);
    sift->detectAndCompute(input, mask, keypoint, descriptor); //descriptor is gonna be overwritten
    bow_extractor->compute(input, keypoint, descriptor);

    if (descriptor.rows > 0 || descriptor.cols > 0) return svm->predict(descriptor);
    else return false;
}

//evaluate IoU
double Detector::evaluate(cv::Mat mask, std::string gt_path) {
    //get ground truth file
    std::string gt = gt_path;
    std::ifstream groundtruth(gt);
    if (!groundtruth.is_open()) { //no ground truth for this image. does not necessarily mean no ships, so skipping to be safe
        std::cout << "Failed to open ground truth file from provided path " + gt_path + ", evaluation aborted." << std::endl;
        return 0;
    }

    //get coordinates of boats from ground truth file
    std::vector<cv::Range> rangex, rangey;
    for (std::string line; std::getline(groundtruth, line); ) {
        //line has format class:X;Y;W;Z;
        std::string x1, x2, y1, y2;
        try {
            line = line.substr(line.find_first_of("0123456789"));
            x1 = line.substr(0, line.find(";"));
            line = line.substr(x1.length() + 1);
            x2 = line.substr(0, line.find(";"));
            line = line.substr(x2.length() + 1);
            y1 = line.substr(0, line.find(";"));
            line = line.substr(y1.length() + 1);
            y2 = line.substr(0, line.find(";"));
        }
        catch (std::exception e) {
            std::cout << "Error parsing GT file (maybe wrong format?)" << std::endl;
            std::cout << "Exception details:" << e.what() << std::endl;
            return false;
        }

        try {
            rangex.push_back(cv::Range(std::stoi(x1), std::stoi(x2)));
            rangey.push_back(cv::Range(std::stoi(y1), std::stoi(y2)));
        }
        catch (std::exception e) {
            std::cout << "Error accepting coordinates from GT file" << std::endl;
            std::cout << "I parsed: " << x1 << " " << x2 << " " << y1 << " " << y2 << std::endl;
            std::cout << "Exception details:" << e.what() << std::endl;
            return false;
        }
    }

    //build real mask
    cv::Mat realmask = cv::Mat::zeros(mask.size(), CV_8U);
    for (int i = 0; i < rangex.size(); i++) {
        realmask(cv::Range(rangey.at(i)), cv::Range(rangex.at(i))) = 255;
    }

    //evaluation
    cv::Mat I;
    cv::Mat U;

    cv::bitwise_and(realmask, mask, I); //intersection
    cv::bitwise_or(realmask, mask, U); //union

    //cv::imshow("Real", realmask);
    //cv::imshow("Detected", mask);
    //cv::imshow("I", I);
    //cv::imshow("U", U);
    //cv::waitKey();

    int i_area = cv::countNonZero(I);
    int u_area = cv::countNonZero(U);
    return (double)i_area / (double)u_area;
}