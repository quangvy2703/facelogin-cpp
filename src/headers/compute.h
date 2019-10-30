//
// Created by liem on 28/10/2019.
//

#ifndef FACIALRECOGNITION_COMPUTE_H
#define FACIALRECOGNITION_COMPUTE_H
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>

cv::Rect RectDlib2CV(dlib::rectangle rect);
std::vector<cv::Rect> ListRectDlib2CV(std::vector<dlib::rectangle> rect);

#endif //FACIALRECOGNITION_COMPUTE_H
