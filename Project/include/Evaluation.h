#pragma once

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
class Evaluation
{
public:
    struct EvaluationData {

        int fp = 0; //false positive
        int fn = 0; //false negative
        int tp = 0; //true positive
        int tn = 0; //true negative

        //Recall = TruePositives / (TruePositives + FalseNegatives)
        float recall = 0;
        //Accuracy = TruePositives / (TruePositives + FalsePositives)
        float accuracy = 0;

    };

    static EvaluationData computeEvaluation(const cv::Mat& mask, const cv::Mat& trueMask);

};

