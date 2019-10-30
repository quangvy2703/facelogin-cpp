#pragma once

#define PI 3.14159265
#include<stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <mxnet/c_predict_api.h>
#include <math.h>
#include <chrono>
#include <ctime>
#include <dirent.h>

#include "headers/face_align.hpp"
#include "headers/mxnet_mtcnn.hpp"
#include "headers/feature_extract.hpp"
//#include "feature_extract.cpp"
#include "headers/comm_lib.hpp"
#include "headers/CONFIGURE.h"

using namespace CONFIG;
using namespace cv;




double ang(cv::Point v1, cv::Point v2) {
    int cosang = v1.x*v2.x + v1.y*v2.y;
    int sinang = v1.x*v2.y - v2.x*v1.y;
    sinang = abs(sinang);
    return atan2(sinang, cosang)*180/PI;
}

int count(bool** matrix) {
    int c = 0;
    for (int i = 0; i < 181;i++){
        for (int j = 0; j < 181; j++)
            c += (int)matrix[i][j];
    }
    return c;
}

void check_ang(cv::Point nose, cv::Point center, cv::Mat frame, bool** registation_faces) {
    cv::Point _nose;
    _nose.x = nose.x-center.x;
    _nose.y = nose.y-center.y;
    double ang1 = ang(_nose,cv::Point(10,0));
    double ang2 = ang(_nose, cv::Point(0,10));

    int rad1 = (int)ang1;
    int rad2 = (int)ang2;
    if (!registation_faces[rad1][rad2]) {
        registation_faces[rad1][rad2] = true;

        if (!registation_faces[(rad1+1)%181][(rad2+1)%181]) {
            registation_faces[(rad1 + 1) % 181][(rad2 + 1) % 181] = true;

        }
        if (!registation_faces[abs(rad1-1)%181][abs(rad2-1)%181]) {
            registation_faces[abs(rad1 - 1) % 181][abs(rad2 - 1) % 181] = true;

        }
        cv::imwrite("../images/"+std::to_string(rad1)+"_"+std::to_string(rad2)+".jpg",frame);
    }
}

bool check(cv::Point p1, cv::Point p2, float error){
    if (abs(p1.x - p2.x) < error && abs(p1.y - p2.y) < error)
        return true;
    return false;
}


void countdown(unsigned int time, cv::VideoCapture cap){
    cv::Mat frame, saved_frame;
    unsigned int count = time;
    auto start = std::chrono::system_clock::now();
    while (true){
        cap >> frame;
        flip(frame, frame, 1);
        saved_frame = frame.clone();

        if (count > 0){
            std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now() - start;
            double elapse = (double) elapsed_seconds.count();
            if (elapse > 1.0){
                start = std::chrono::system_clock::now();
                count -= 1;
            }
        }
        else{
            break;
        }

        int h = frame.rows;
        int w = frame.cols;

        ellipse(frame, Point((int)w/2, (int)h/2 ), Size(CONFIG::ELLIPSE_WIDTH, CONFIG::ELLIPSE_HEIGHT),
                0, 0, 360, CONFIG::BLACK, -1);

        cv::add(frame, saved_frame, frame);

        putText(frame, "Start register after " + std::to_string(count) + "s", Point((int)w/2 - 150, (int)h/2), cv::FONT_HERSHEY_SIMPLEX,
                1, CONFIG::TEAL, 2, cv::LINE_AA);

        cv::imshow("frame", frame);
		cv::waitKey(1);
    }
}

void get_nose(cv::VideoCapture cap, MxNetMtcnn& mtcnn){
    bool pass = false;
    cv::Mat frame, saved_frame;
    cap >> frame;

    int h = frame.rows;
    int w = frame.cols;

    CONFIG::W = w;
    CONFIG::H = h;
    Point start = Point((int)w/2, (int)h/2);

    cv::Point P_LEFT = cv::Point(int(w / 2 - CONFIG::ELLIPSE_WIDTH), int(h / 2));
    cv::Point P_RIGHT = cv::Point(int(w / 2 + CONFIG::ELLIPSE_WIDTH), int(h / 2));
    cv::Point P_TOP = cv::Point(int(w / 2), int(h / 2 - CONFIG::ELLIPSE_HEIGHT));
    cv::Point P_BOTTOM = cv::Point(int(w / 2), int(h / 2 + CONFIG::ELLIPSE_HEIGHT));
    while (true){
        cap >> frame;
        flip(frame, frame, 1);
        saved_frame = frame.clone();
        circle(frame, start, 1, CONFIG::BLUE, 10);
        ellipse(frame, start, Size(CONFIG::ELLIPSE_WIDTH, CONFIG::ELLIPSE_HEIGHT), 0, 0, 360, CONFIG::BLACK, -1);
        cv::GaussianBlur(frame, frame, cv::Size(15, 15), BORDER_DEFAULT);
        cv::add(frame, saved_frame, frame);


//        cout << P_LEFT << P_RIGHT << endl;
//        cv::line(frame, CONFIG::P_LEFT, CONFIG::P_RIGHT, CONFIG::TEAL, 10);
//        cv::line(frame, CONFIG::P_TOP, CONFIG::P_BOTTOM, CONFIG::TEAL, 10);

        cv::line(frame, P_LEFT, P_RIGHT, CONFIG::TEAL, 10);
        cv::line(frame, P_TOP, P_BOTTOM, CONFIG::TEAL, 10);

        std::vector<face_box> face_info;

	//detect with mtcnn, get detect reuslt with bounding box and landmark point
	    mtcnn.Detect(saved_frame, face_info);
	    for (int i = 0; i < face_info.size(); i++){
	        cv::Point nose(face_info[i].landmark.x[2], face_info[i].landmark.y[2]);
	        vector<cv::Point> pts;
	        pts.push_back(P_TOP); pts.push_back(nose); pts.push_back(P_BOTTOM);
	        cv::polylines(frame, pts, false, CONFIG::RED, 3);

            pts.clear();

	        pts.push_back(P_LEFT); pts.push_back(nose); pts.push_back(P_RIGHT);
	        cv::polylines(frame, pts, false, CONFIG::RED, 3);

	        if (check(start, nose, 5.0) == 1)
	            return;
        }

        cv::imshow("frame", frame);
		cv::waitKey(1);
    }
}

void get_faces(cv::VideoCapture cap, MxNetMtcnn& mtcnn, Mxnet_extract& extract){

    bool** registation_faces = new bool*[181];

    for (int i = 0; i < 181; i++) {
        registation_faces[i] = new bool[181];
        for (int j = 0; j < 181;j++) {
            registation_faces[i][j] = false;
        }
    }



    cv::Mat frame, saved_frame;
    cap >> frame;

    int h = frame.rows;
    int w = frame.cols;

    CONFIG::W = w;
    CONFIG::H = h;
    Point start = Point((int)w/2, (int)h/2);

    cv::Point P_LEFT = cv::Point(int(w / 2 - CONFIG::ELLIPSE_WIDTH), int(h / 2));
    cv::Point P_RIGHT = cv::Point(int(w / 2 + CONFIG::ELLIPSE_WIDTH), int(h / 2));
    cv::Point P_TOP = cv::Point(int(w / 2), int(h / 2 - CONFIG::ELLIPSE_HEIGHT));
    cv::Point P_BOTTOM = cv::Point(int(w / 2), int(h / 2 + CONFIG::ELLIPSE_HEIGHT));
    cv::Mat pivot(cv::Size(128, 1), CV_64FC1);
    int idx = 0;
    while (true){
        cap >> frame;
        flip(frame, frame, 1);
        saved_frame = frame.clone();
        ellipse(frame, start, Size(CONFIG::ELLIPSE_WIDTH, CONFIG::ELLIPSE_HEIGHT), 0, 0, 360, CONFIG::BLACK, -1);
        cv::GaussianBlur(frame, frame, cv::Size(15, 15), BORDER_DEFAULT);
        cv::add(frame, saved_frame, frame);
        cv::circle(frame, start, 1, CONFIG::BLUE, 10);
        std::vector<face_box> face_info;
        mtcnn.Detect(saved_frame, face_info);
        cv::Point nose;
	    for (int i = 0; i < face_info.size(); i++){
	        nose = cv::Point(face_info[i].landmark.x[2], face_info[i].landmark.y[2]);
	        vector<cv::Point> pts;
	        pts.push_back(P_TOP); pts.push_back(nose); pts.push_back(P_BOTTOM);
	        cv::polylines(frame, pts, false, CONFIG::GREEN, 3);

            pts.clear();

	        pts.push_back(P_LEFT); pts.push_back(nose); pts.push_back(P_RIGHT);
	        cv::polylines(frame, pts, false, CONFIG::GREEN, 3);

            Rect crop((int)face_info[i].x0, (int)face_info[i].y0,
                    (int)(face_info[i].x1 - face_info[i].x0),
                    (int)(face_info[i].y1 - face_info[i].y0));
//            cout << face_info[i].x0 << " " << face_info[i].y0
//                    << " " << face_info[i].x1 << " " << face_info[i].y1 << endl;
            cv::Mat cropped = saved_frame(crop);

            cv::Mat features = get_features(face_info, extract, saved_frame);
            cout << "Features size " << features.size() << endl;
            features.convertTo(features, CV_64FC1);
            cv::add(pivot, features, pivot);
        }

        check_ang(nose, start, saved_frame, registation_faces);
        int no = count(registation_faces);


        int percent = (int) (1.0 * no / CONFIG::MAX_IMAGES * 100);


        if (percent < 25)
            putText(frame, "Registration " + std::to_string(percent), Size(20, 20), cv::FONT_HERSHEY_SIMPLEX, 1,
                                CONFIG::RED, 2, cv::LINE_AA);
        else if (percent < 50)
            putText(frame, "Registration " + std::to_string(percent), Size(20, 20), cv::FONT_HERSHEY_SIMPLEX, 1,
                                CONFIG::ORANGE, 2, cv::LINE_AA);
        else if (percent < 80)
            putText(frame, "Registration " + std::to_string(percent), Size(20, 20), cv::FONT_HERSHEY_SIMPLEX, 1,
                                CONFIG::TEAL, 2, cv::LINE_AA);
        else
            putText(frame, "Registration " + std::to_string(percent), Size(20, 20), cv::FONT_HERSHEY_SIMPLEX, 1,
                                CONFIG::GREEN, 2, cv::LINE_AA);

        if (percent >= 100){
            pivot.convertTo(pivot, CV_32FC1);
            pivot = pivot / CONFIG::MAX_IMAGES;
            cv::FileStorage fs("../features.xml", cv::FileStorage::WRITE);
            fs << "features" << pivot;
            fs.release();
            return;
        }

        cv::imshow("frame", frame);
		cv::waitKey(1);

    }
}



void registration()
{
	MxNetMtcnn mtcnn;
	mtcnn.LoadModule("../mtcnn_model");

	Mxnet_extract extract;
	extract.LoadExtractModule("../feature_model/model-0000.params", "../feature_model/model-symbol.json", 1, 3, 112, 112);


	cv::VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
		return;

    cv::Mat frame;

    // Start countdown
    cout << "Start countdown" << endl;
    countdown(1, cap);
    cout << "Start get nose" << endl;
    get_nose(cap, mtcnn);
    cout << "Start get faces" << endl;
    get_faces(cap, mtcnn, extract);


	//loading features
	cv::FileStorage fs("../features.xml", cv::FileStorage::READ);
	cv::Mat features;
	fs["features"] >> features;

	cap.release();
}

int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

void run()
{
	MxNetMtcnn mtcnn;
	mtcnn.LoadModule("../mtcnn_model");

	Mxnet_extract extract;
	extract.LoadExtractModule("../feature_model/model-0000.params", "../feature_model/model-symbol.json", 1, 3, 112, 112);


	cv::VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
		return;

    cv::Mat frame;

	//loading features
	cv::FileStorage fs("../features.xml", cv::FileStorage::READ);
	cv::Mat features;
	fs["features"] >> features;


	while (1)
	{
		cap >> frame;
		flip(frame, frame, 1);
		double start = static_cast<double>(cv::getTickCount());

		vector<cv::Rect> face_boxes = recognition(mtcnn, extract, frame, features);
		double time = ((double)cv::getTickCount() - start) / cv::getTickFrequency();
		std::cout << "Total time: " << time << "s " << std::endl;

		if (frame.empty())
			break;
		cv::imshow("frame", frame);
		cv::waitKey(1);

	}
	cap.release();
}


void test()
{
	MxNetMtcnn mtcnn;
	mtcnn.LoadModule("../mtcnn_model");



	Mxnet_extract extract;
	extract.LoadExtractModule("../feature_model/model-0000.params", "../feature_model/model-symbol.json", 1, 3, 112, 112);

	//loading features
	cv::FileStorage fs("../features.xml", cv::FileStorage::READ);
	cv::Mat features;
	fs["features"] >> features;

	//loading labels
	std::ifstream file("labels.txt");
	std::string t;
	while (std::getline(file, t)) {}

	std::vector<std::string> labels;
	SplitString(t, labels, " ");

	cv::VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
		return;
    cv::Mat frame;

    string dir = string("/media/vy/DATA/data/All-Age-Faces Dataset/original images/");
    vector<string> files = vector<string>();

    cout << "Get dir " << endl;
    getdir(dir,files);

    for (unsigned int i = 10;i < files.size() - 10;i++) {

        cout << files[i] << endl;
		frame = cv::imread(dir + files[i]);
		double start = static_cast<double>(cv::getTickCount());


		recognition(mtcnn, extract, frame, features);
		double time = ((double)cv::getTickCount() - start) / cv::getTickFrequency();
		std::cout << "spent: " << time << "s " << std::endl;

//		if (frame.empty())
//			break;
		cv::imshow("frame", frame);
		cv::waitKey(1);

	}
	cap.release();
}



int main(int argc, char* argv[]) {

//	registration();
    run();
	system("pause");
	return 0;
}
