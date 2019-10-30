//#include <dlib/opencv.h>
//#include <opencv2/highgui/highgui.hpp>
//#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>
//#include <dlib/image_processing.h>
//#include <dlib/gui_widgets.h>
////int main() {
//   VideoCapture cap(0);
//   if (!cap.isOpened())
//   {
//       cout<<"Error opening video stream"<< endl;
//       return -1;
//   }
////    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
////    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
//   while (1) {
//       Mat frame;
//       cap >> frame;
//       if (frame.empty())
//           break;
//       imshow("frame", frame);
//       char c=(char)waitKey(25);
//       if (c == 27)
//           break;
//   }
//   cap.release();
//   destroyAllWindows();
//   return 0;
//}