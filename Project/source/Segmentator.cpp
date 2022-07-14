
//MY IMPORTS
#include "../include/Segmentator.h"
#include "../include/Utils.h"

#include <iostream>
/**
* This file represent the Segmentator module
* @author : Daniela Cuza, Simone D'antimo
*/

cv::Mat Segmentator::getSegmentationMaskBW()
{
    //Resize BW mask provided by the model
    cv::Mat resized;        
    cv::resize(bwRawMask, resized, cv::Size(std::get<0>(image).cols, std::get<0>(image).rows), cv::INTER_CUBIC);

    //Threshold the upscalded image
    cv::Mat thresholded;
    cv::threshold(resized, thresholded, HIGHEST_VALUE / 2, HIGHEST_VALUE, cv::THRESH_BINARY);
    
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

void Segmentator::readBWMaskRaw(cv::String pathBWMaskRaw)
{
    //Read the image
    cv::String actualPath = cv::samples::findFile(pathBWMaskRaw);
    cv::Mat imageRead = cv::imread(actualPath, cv::IMREAD_GRAYSCALE);

    //Test if the image is correct
    if (imageRead.empty())
        throw std::invalid_argument("The B&W raw mask provided is not correct !");

    //Save "bwRawMask"
    bwRawMask = imageRead;
}

Segmentator::EvaluationData Segmentator::computePixelAccuracy(const cv::Mat& bwMask)
{
    unsigned char intensity;
    unsigned char intensityTrue;
    EvaluationData maskEvaluation;
    unsigned char tmp = 0;

    //For every pixel calculate false positive, false negatives, true positive, true negatives
    for (int i = 0; i < bwMask.rows; i++)
    {
        for (int j = 0; j < bwMask.cols; j++)
        {
            intensity = bwMask.at<unsigned char>(i, j);
            intensityTrue = groundTruth.at<unsigned char>(i, j);
            if (intensityTrue == HIGHEST_VALUE)
            {
                if (intensity == HIGHEST_VALUE)
                    maskEvaluation.tp++; //both pixel white -> true positive
                else
                    maskEvaluation.fn++; //pixel should be white but is black -> false negative
            }
            else 
            {
                if (intensity == HIGHEST_VALUE)
                    maskEvaluation.fp++; //Pixel should be black, but is white --> false positive
                else
                    maskEvaluation.tn++; // both pixel black --> true negative    
            }
        }
    }

    //Compute recall, precision and pixel accuracy
    maskEvaluation.recall = (maskEvaluation.tp / (maskEvaluation.tp + maskEvaluation.fn));
    
    maskEvaluation.precision = (maskEvaluation.tp / (maskEvaluation.tp + maskEvaluation.fp));
    
    maskEvaluation.pixelAccuracy =  ((maskEvaluation.tn + maskEvaluation.tp) /
        (maskEvaluation.tn + maskEvaluation.tp + maskEvaluation.fn + maskEvaluation.fp));

    return maskEvaluation;
}