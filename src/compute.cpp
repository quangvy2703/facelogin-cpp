////
//// Created by liem on 28/10/2019.
////
//
//#include "headers/compute.h"
//
//cv::Rect RectDlib2CV(dlib::rectangle rect) {
//    cv::Rect det;
//    cv::Rect dets;
//    dets.x = rect.left();
//    dets.y = rect.top();
//    dets.width = rect.right()- dets.x;
//    dets.height = rect.bottom() - dets.y;
//    return dets;
//}
//
//std::vector<cv::Rect> ListRectDlib2CV(std::vector<dlib::rectangle> rects) {
//    std::vector<cv::Rect> dets;
//    for(int i = 0; i < rects.size(); i++) {
//        dets.push_back(RectDlib2CV(rects[i]));
//    }
//    return dets;
//}