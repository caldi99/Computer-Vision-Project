
//MY IMPORTS
#include "../include/Segmentator.h"
#include "../include/Utils.h"

/**
* This file represent the Segmentator module
* @author : Daniela Cuza, Simone D'antimo and Francesco Caldivezzi
*/
/*
void Segmentator::segment_1(cv::String pathImage)
{
	// **********  read the image ************** //
	cv::Mat img = cv::imread(pathImage, cv::IMREAD_COLOR);


    
    // ************** definition of variables ************** //
    cv::Mat out_bf;
    cv::Mat skin_region;
    cv::Mat src;


	// ********** step 1) apply bilateral filter ************** //
    cv::bilateralFilter(img,out_bf, 5, 150, 150);

    
	// ********** step 2) apply threshold *************

    for (int i = 0; i < out_bf.rows; i++) {
        for (int j = 0; j < out_bf.cols; j++) {

            int R = out_bf.at<cv::Vec3b>(i, j)[0];
            int G = out_bf.at<cv::Vec3b>(i, j)[1];
            int B = out_bf.at<cv::Vec3b>(i, j)[2];
            int max_value;
            int min_value;

            // search max value among R, G, B
            if (R >= G && R >= B) {
                max_value = R;
            }
            else if (G >= R && G >= B) {
                max_value = G;
            }
            else {
                max_value = B;
            }

            // search min value among R, G, B
            if (R <= G && R <= B) {
                min_value = R;
            }
            else if (G <= R && G <= B) {
                min_value = G;
            }
            else {
                min_value = B;
            }

            if ((B > 75 && G > 20 && R > 5 && (max_value - min_value > 5) && abs(B - G) > 5 && B > G && B > R) || (B > 180 && G > 180 && R > 130 && abs(B - G) <= 35 && B > R && G > R)) {
                
                out_bf.at<cv::Vec3b>(i, j)[0] = R;
                out_bf.at<cv::Vec3b>(i, j)[1] = G;
                out_bf.at<cv::Vec3b>(i, j)[2] = B;
            }
            else {
                out_bf.at<cv::Vec3b>(i, j)[0] = 0;
                out_bf.at<cv::Vec3b>(i, j)[1] = 0;
                out_bf.at<cv::Vec3b>(i, j)[2] = 0;
            }
        }
    }
    
    
            
    cv::cvtColor(out_bf, out_bf, cv::COLOR_BGR2YCrCb); // covert from BGR to YCrCb
    cv::inRange(out_bf, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), skin_region); //compute mask
    out_bf.copyTo(src, skin_region); //apply mask
    cv::cvtColor(src, src, cv::COLOR_YCrCb2BGR); // covert from YCrCb TO BGR
    
    
    cv::imshow("Image after bilateral filter and threshold", src);

    
    
}*/

cv::Mat Segmentator::getSegmentationMaskBW()
{
    //Read Model
    cv::dnn::Net network = cv::dnn::readNetFromONNX(pathModel);

    //Set input
    network.setInput(cv::dnn::blobFromImage(std::get<0>(image), 1.0, cv::Size(WIDTH_INPUT_CNN, HEIGHT_INPUT_CNN), true, false));

    //Forward
    std::vector<cv::Mat> output = network.forward();
    
    //Get Raw Mask
    cv::Mat rawMaskBW = convertOutputCNNToBWMask(output.at(0));

    //Resize the Raw Mask
    cv::Mat rawMaskResized;
    cv::resize(rawMaskBW, rawMaskResized,cv::Size(std::get<0>(image).cols, std::get<0>(image).rows), cv::INTER_CUBIC);

    //Threshold the resized image
    cv::Mat thresholded;
    cv::threshold(rawMaskResized, thresholded, 1, HIGHEST_VALUE, cv::THRESH_BINARY);
    
    return thresholded;    
}

cv::Mat Segmentator::getImageWithSegmentations(const cv::Mat& bwMask)
{
    //TODO : APPLY DILATION EROSION BEFORE AND AFTER ??

    //Get Connected components of the B&W image
    cv::Mat labelImage(bwMask.rows, bwMask.cols, CV_32S);
    int nLabels = cv::connectedComponents(bwMask, labelImage, 8, CV_32S);
    
    //Create random colors for coloring the hands in the image
    std::vector<cv::Vec3b> colors(nLabels);    
    
    //Background
    colors.at(0) = cv::Vec3b(0, 0, 0); 
    
    //Random Colors
    for (int i = 1; i < colors.size(); i++)    
        colors.at(i) = cv::Vec3b((std::rand() % HIGHEST_VALUE), (std::rand() % HIGHEST_VALUE), (std::rand() % HIGHEST_VALUE));
    
    //Create mask with colored components
    cv::Mat colorMask(bwMask.rows, bwMask.cols, CV_8UC3);
    for (int r = 0; r < colorMask.rows; r++)
        for (int c = 0; c < colorMask.cols; c++)       
            colorMask.at<cv::Vec3b>(r, c) = colors.at(labelImage.at<int>(r, c));

    //Create image with colored hands
    cv::Mat ret(bwMask.rows, bwMask.cols, CV_8UC3);
    for(int r = 0; r < ret.rows; r++)
        for (int c = 0; c < ret.cols; c++)
            if (colorMask.at<cv::Vec3b>(r, c) == colors.at(0)) // If background then pixel orginal image            
                ret.at<cv::Vec3b>(r, c) = std::get<0>(image).at<cv::Vec3b>(r, c);
            else 
                ret.at<cv::Vec3b>(r, c) = colorMask.at<cv::Vec3b>(r, c);

    return ret;
}

void Segmentator::savePixelAccuracies(cv::String outputPath, const cv::Mat& bwMask)
{
    //Get ground truths of the image
    std::ofstream file(outputPath + std::get<1>(image) + ".txt");

    //Compute Evaluations
    EvaluationData evalutation = computePixelAccuracy(bwMask);

    //Write Evaluations
    file << "TRUE POSIVE : " << evalutation.tp << std::endl <<
        "TRUE NEGATIVE : " << evalutation.tn << std::endl <<
        "FALSE POSIVE : " << evalutation.fp << std::endl <<
        "FALSE NEGATIVE : " << evalutation.tn << std::endl <<
        "PRECISION : " << evalutation.precision << std::endl <<
        "RECALL : " << evalutation.recall << std::endl <<
        "PIXEL ACCURACY : " << evalutation.pixelAccuracy << std::endl;

    //Close File
    file.close();
}

void Segmentator::saveSegmentations(cv::String pathSegmentation, const cv::Mat& bwMask)
{
    //Construct name of the image that we are going to save
    cv::String nameFileExtension = std::get<1>(image) + "_segmentations.jpg";

    //Get Image Segmented
    cv::Mat imageToSave = getImageWithSegmentations(bwMask);

    //Save Image
    cv::imwrite(pathSegmentation + nameFileExtension, imageToSave);
}

void Segmentator::saveSegmentationMaskBW(cv::String pathSegmentationMaskBW, const cv::Mat& bwMask)
{
    //Construct name of the image that we are going to save
    cv::String nameFileExtension = std::get<1>(image) + "_bwmask.jpg";

    //Save Image
    cv::imwrite(pathSegmentationMaskBW + nameFileExtension, bwMask);
}

void Segmentator::readImage(cv::String pathImage)
{
    //Read the image
    cv::String actualPath = cv::samples::findFile(pathImage);
    cv::Mat imageRead = cv::imread(actualPath);

    //Test if the image is correct
    if (imageRead.empty())
        throw std::invalid_argument("The image provided is not correct !");

    //Get name of the image
    std::vector<cv::String> parts = Utils::split(pathImage, '/');
    cv::String name = Utils::split(parts.at(parts.size() - 1), '.').at(0);

    //Save "image"
    image = std::make_tuple(imageRead, name);
}

void Segmentator::readGroundTruth(cv::String pathGroundTruth)
{
    //Read the image
    cv::String actualPath = cv::samples::findFile(pathGroundTruth);
    cv::Mat imageRead = cv::imread(actualPath, cv::IMREAD_GRAYSCALE);

    //Test if the image is correct
    if (imageRead.empty())
        throw std::invalid_argument("The Ground Truth mask provided is not correct !");

    //Save "ground truth"
    groundTruth = imageRead;
}

void Segmentator::setModel(cv::String pathModel)
{
    this->pathModel = pathModel;
}

cv::Mat Segmentator::convertOutputCNNToBWMask(const cv::Mat& outputCNN)
{
    cv::Mat ret(outputCNN.rows, outputCNN.cols, CV_8UC1);
    
    //Convert probabilities into pixels Black and White
    for (int r = 0; r < outputCNN.rows; r++)    
        for (int c = 0; c < outputCNN.cols; c++)        
            if (outputCNN.at<float>(r, c) > THRESHOLD_CNN)
                ret.at<unsigned char>(r, c) = HIGHEST_VALUE;
            else 
                ret.at<unsigned char>(r, c) = 0;
    
    return ret;
}

Segmentator::EvaluationData Segmentator::computePixelAccuracy(const cv::Mat& bwMask)
{
    unsigned char intensity;
    unsigned char intensityTrue;
    EvaluationData maskEvaluation;
    unsigned char tmp = 0;

    //For every pixel calculate false positive, false negatives, true positive, true negatives
    for (int i = 0; i < bwMask.rows; i++)    
        for (int j = 0; j < bwMask.cols; j++) 
        {
            intensity = bwMask.at<unsigned char>(i, j);
            intensityTrue = groundTruth.at<unsigned char>(i, j);
            if (intensityTrue == HIGHEST_VALUE)
                if (intensity == HIGHEST_VALUE)
                    maskEvaluation.tp++; //both pixel white -> true positive
                else
                    maskEvaluation.fn++; //pixel should be white but is black -> false negative
            
            else             
                if (intensity == HIGHEST_VALUE)
                    maskEvaluation.fp++; //Pixel should be black, but is white --> false positive
                else
                    maskEvaluation.tn++; // both pixel black --> true negative    
            
        }    

    //Compute recall, precision and pixel accuracy
    maskEvaluation.recall = static_cast<float> (maskEvaluation.tp / (maskEvaluation.tp + maskEvaluation.fn));
    
    maskEvaluation.precision = static_cast<float>(maskEvaluation.tp / (maskEvaluation.tp + maskEvaluation.fp));
    
    maskEvaluation.pixelAccuracy = static_cast<float> ((maskEvaluation.tn + maskEvaluation.tp) /
        (maskEvaluation.tn + maskEvaluation.tp + maskEvaluation.fn + maskEvaluation.fp));

    return maskEvaluation;
}