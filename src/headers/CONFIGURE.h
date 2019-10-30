#pragma once

namespace CONFIG
{
    int W, H;
    int MAX_IMAGES = 350;
    bool _USE_GRAY = true;
    bool _USE_BRIGHTNESS = true;
    bool _USE_NORM = true;
    bool _USE_FLIP = true;
    bool _REGISTER = false;
    int ELLIPSE_WIDTH = 170;
    int ELLIPSE_HEIGHT = 220;
    cv::Scalar BLACK = cv::Scalar(0, 0, 0);
    cv::Scalar TEAL = cv::Scalar(255, 255, 0);
    cv::Scalar BLUE = cv::Scalar(181, 212, 66);
    cv::Scalar RED = cv::Scalar(0, 0, 0255);
    cv::Scalar GREEN = cv::Scalar(0, 255, 0);
    cv::Scalar ORANGE = cv::Scalar(122, 170, 255);

}
