
#include "../include/UtilsHoughTransform.h"



UtilsHoughTransform::UtilsHoughTransform()
{
}

void UtilsHoughTransform::generalizedHoughTransform(cv::Mat pattern, cv::Mat image)
{

    cv::Ptr<cv::GeneralizedHough> alg;

    
    cv::Ptr<cv::GeneralizedHoughGuil> guil = cv::createGeneralizedHoughGuil();

    guil->setMinDist(100);
    guil->setLevels(360);
    guil->setDp(2);
    guil->setMaxBufferSize(1000);

    guil->setMinAngle(0);
    guil->setMaxAngle(360);
    guil->setAngleStep(1);
    guil->setAngleThresh(10000);

    guil->setMinScale(0.5);
    guil->setMaxScale(2);
    guil->setScaleStep(0.05);
    guil->setScaleThresh(1000);

    guil->setPosThresh(100);

    alg = guil;
    

    std::vector<cv::Vec4f> position;


    cv::Mat patternGray;
    cv::cvtColor(pattern, patternGray, cv::COLOR_BGR2GRAY);

    cv::imshow("keypoints", patternGray);
    cv::waitKey(0);

    cv::Mat imageGray;
    cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);



    try {
        alg->setTemplate(patternGray);
    }
    catch (cv::Exception e)
    {

    }
    



    alg->detect(imageGray, position);


    
    cv::Mat out;
    cv::cvtColor(image, out, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < position.size(); ++i)
    {
        cv::Point2f pos(position[i][0], position[i][1]);
        float scale = position[i][2];
        float angle = position[i][3];

        cv::RotatedRect rect;
        rect.center = pos;
        rect.size = cv::Size2f(pattern.cols * scale, pattern.rows * scale);
        rect.angle = angle;

        cv::Point2f pts[4];
        rect.points(pts);

        line(out, pts[0], pts[1], cv::Scalar(0, 0, 255), 3);
        line(out, pts[1], pts[2], cv::Scalar(0, 0, 255), 3);
        line(out, pts[2], pts[3], cv::Scalar(0, 0, 255), 3);
        line(out, pts[3], pts[0], cv::Scalar(0, 0, 255), 3);
    }

    cv::imshow("out", out);
    cv::waitKey();


}
