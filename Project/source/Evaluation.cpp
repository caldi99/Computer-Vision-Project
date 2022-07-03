//Author: Simone D'Antimo
#include "../include/Evaluation.h"

using namespace cv;

/* Given the mask calculated and the true mask this function count all the true/false positive/negative pixel
*  and calculates also the accuracy and recall.
*  Important: Both mask must have only blackand white pixels
* @parameters:
*  mask : the mask guessed by our application
*  trueMask: the true mask of the image
* @return: An Evaluation data structure containing all the information on true/false positive/negative
*          recall and accuracy
*/

Evaluation::EvaluationData Evaluation::computeEvaluation(const Mat& mask, const Mat& trueMask) {

    uchar intensity;
    uchar intensityTrue;
    EvaluationData maskEvaluation;
    uchar tmp = 0;

    //Input image must be 1 channel grayScale
    if (mask.channels() > 1) {
        cvtColor(mask, mask, COLOR_BGR2GRAY);
    }

    if (trueMask.channels() > 1) {
        cvtColor(trueMask, trueMask, COLOR_BGR2GRAY);
    }

    //For every pixel calculate false positive, false negatives, true positive, true negatives
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            intensity = mask.at<uchar>(i, j);
            intensityTrue = trueMask.at<uchar>(i, j);

            if (intensityTrue == 255) {

                if (intensity == 255)
                    maskEvaluation.tp++; //both pixel white -> true positive
                else
                    maskEvaluation.fn++; //pixel should be white but is black -> false negative
            }

            else {
                if (intensity == 255)
                    maskEvaluation.fp++; //Pixel should be black, but is white --> false positive
                else
                    maskEvaluation.tn++; // both pixel black --> true negative    
            }
        }
    }

    //calculate recall and accuracy
    maskEvaluation.recall = (float)maskEvaluation.tp / (maskEvaluation.tp + maskEvaluation.fn);
    maskEvaluation.accuracy = (float)maskEvaluation.tp / (maskEvaluation.tp + maskEvaluation.fp);

    return maskEvaluation;
}
