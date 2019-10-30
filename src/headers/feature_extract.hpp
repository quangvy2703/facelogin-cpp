#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <stdlib.h>


#include "mxnet/c_predict_api.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "comm_lib.hpp"
#include "buffer_file.hpp"
#include "mxnet_mtcnn.hpp"
#include "CONFIGURE.h"


using namespace std;
using namespace mxnet::cpp;
using namespace cv;
using namespace CONFIG;


struct class_info{
	double min_distance;
	int index;
};

struct verification_info{
    double distance;
    double similarity;
};



double sigmoid(const double x){
    return 1 / (1 + exp(-x));
};

double compute_distance(const cv::Mat& img, const cv::Mat& pivot){
    int rows = pivot.rows;
    cv::Mat broad;
    cv::repeat(img, rows, 1, broad);

    broad = pivot - broad;
    cv::pow(broad, 2, broad);
    cv::reduce(broad, broad, 1, cv::REDUCE_AVG);

    double distance = broad.at<float>(0, 0);
//    cout << "Before sqrt " << distance << endl;
    distance = sqrt(distance);
//    cout << "After sqrt " << distance << endl;
//
//    distance = log(distance) * log(distance) * log(distance);
//    cout << "After expand " << distance << endl;


    return distance;

};

double compute_similarity(const cv::Mat& img, const cv::Mat& pivot){
//    int rows = pivot.rows;
//    cv::Mat broad;
//    cv::repeat(img, rows, 1, broad);

//    cout << img.size() << " " << pivot.size() << endl;
    double dot_prod = img.dot(pivot);
//    for (int i=0; i < (int)broad.rows; i++){
//        dot_prod = dot_prod + broad.at<double>(i, 0) * pivot.at<double>(i, 0);
//    }

//    cout << "Before expand " << dot_prod << endl;
    double expand = dot_prod;
//    double expand = (dot_prod * dot_prod * dot_prod);
//    cout << "After abs " << expand << endl;
////    expand = sigmoid(expand);
////    expand = sigmoid(log(expand));
//    cout << "After expand " << expand << endl;

    return expand;

};

double expand_different(const double score, const int time){
    double expand = 1;
    for (int i = 0; i < time; i++){
        expand = expand * log(score);
    }
    return expand;
};


class_info classify(const cv::Mat& img, const  cv::Mat& cmp)
{
	int rows = cmp.rows;
	cv::Mat broad;
	cv::repeat(img, rows, 1, broad);

	broad = broad - cmp;
	cv::pow(broad,2,broad);
	cv::reduce(broad, broad, 1, cv::REDUCE_SUM);

	double dis;
	cv::Point point;
	cv::minMaxLoc(broad, &dis, 0, &point, 0);

	return class_info{dis, point.y};
};

verification_info verify(const cv::Mat& img, const cv::Mat& pivot){
    double distance = compute_distance(img, pivot);
    double similarity = compute_similarity(img, pivot);
//    similarity = expand_different(similarity, 3);
//    distance = expand_different(distance, 3);
//    double similarity = 0;
    return verification_info{distance, similarity};
};


cv::Mat increase_brightness(cv::Mat img, const int threshold=200){
    cv::Mat gray;
    cv::cvtColor(img, gray, COLOR_BGR2GRAY);
    float brightness = cv::mean(gray)[0];
    float value = 1.2;
    bool increase = true;
    if (brightness > threshold){
        value = 0.2;
        increase = false;
    }


    cv::Mat hsv, return_img;
    return_img = img.clone();
    int count = 0;
    while (true){
        count += 1;
        cv::cvtColor(return_img, hsv, COLOR_BGR2HSV);
        vector<Mat> hsv_planes;
        split( hsv, hsv_planes);
//        hsv_planes[0] // H channel
//        hsv_planes[1] // S channel
        cv::Mat v = hsv_planes[2]; // V channel
//        cout << v.at<uint8_t>(490, 2) << endl;
        for ( int w = 0; w < v.cols; w++){
            for ( int h = 0; h < v.rows; h++){
                if (v.at<uint8_t>(h, w) * value < 255)
                    v.at<uint8_t>(h, w) =  v.at<uint8_t>(h, w) * value;
                else
                    v.at<uint8_t>(h, w) = 255;
            }
        }

        hsv_planes[2] = v;
        cv::merge(hsv_planes, return_img);

        cv::cvtColor(return_img, return_img, COLOR_HSV2BGR);

        cv::cvtColor(return_img, gray, COLOR_BGR2GRAY);
//        cout << cv::mean(gray)[0] << endl;
        if ((float)cv::mean(gray)[0] > threshold && increase){
//                cout << "Here 1 " << cv::mean(gray)[0] << endl;
                return return_img;
        }
        if ((float)cv::mean(gray)[0] < threshold && !increase){
//                cout << "Here 2 " << cv::mean(gray)[0] << endl;
                return return_img;
        }

        if (count == 20){
            return return_img;
        }

    }
    return return_img;
}

cv::Mat get_gray_image(cv::Mat img){
    cv::Mat _img, return_img;
    cv::cvtColor(img, _img, COLOR_BGR2GRAY);
    cv::cvtColor(_img, return_img, COLOR_GRAY2BGR);
    return return_img;
}

cv::Mat norm_img(cv::Mat img){
    cv::Mat return_img(112, 112, CV_32FC3);
    cv::normalize(img, return_img, 0, 255, NORM_MINMAX, CV_8UC1);
    return return_img;
}




class Mxnet_extract
{
public:
	~Mxnet_extract()
	{
		if(pred_feature)
		    MXPredFree(pred_feature);
	}
	int LoadModel(const std::string & fname, std::vector<char>& buf)
	{
		std::ifstream fs(fname, std::ios::binary | std::ios::in);

		if (!fs.good())
		{
			std::cerr << fname << " does not exist" << std::endl;
			return -1;
		}

		fs.seekg(0, std::ios::end);
		int fsize = fs.tellg();

		fs.seekg(0, std::ios::beg);
		buf.resize(fsize);
		fs.read(buf.data(), fsize);

		fs.close();

		return 0;

	}

	int LoadExtractModule(const std::string& param_file, const std::string& json_file,
		int batch, int channel, int input_h, int input_w)
	{

		std::vector<char> param_buffer;
		std::vector<char> json_buffer;

		if (LoadModel(param_file, param_buffer)<0)
			return -1;

		if (LoadModel(json_file, json_buffer)<0)
			return -1;

		int device_type = 1;
		int dev_id = 0;
		mx_uint  num_input_nodes = 1;
		const char * input_keys[1];
		const mx_uint input_shape_indptr[] = { 0, 4 };
		const mx_uint input_shape_data[] = {
			static_cast<mx_uint>(batch),
			static_cast<mx_uint>(channel),
			static_cast<mx_uint>(input_h),
			static_cast<mx_uint>(input_w)
		};

		input_keys[0] = "data";

		int ret = MXPredCreate(json_buffer.data(),
			param_buffer.data(),
			param_buffer.size(),
			device_type,
			dev_id,
			num_input_nodes,
			input_keys,
			input_shape_indptr,
			input_shape_data,
			&pred_feature
		);

		return ret;
	}


	cv::Mat extractFeature(const cv::Mat& img)
	{

		int width = img.cols;
		int height = img.rows;

        cv::Mat img_rgb(height, width, CV_32FC3);
		img.convertTo(img_rgb, CV_32FC3);
		cv::cvtColor(img_rgb, img_rgb, cv::COLOR_BGR2RGB);

		std::vector<float> input(3 * height * width);
		std::vector<cv::Mat> input_channels;

		set_input_buffer(input_channels, input.data(), height, width);
		cv::split(img_rgb, input_channels);


		MXPredSetInput(pred_feature, "data", input.data(), input.size());
		MXPredForward(pred_feature);

		mx_uint *shape = NULL;
		mx_uint shape_len = 0;

		MXPredGetOutputShape(pred_feature, 0, &shape, &shape_len);

		int feature_size = 1;
		for (unsigned int i = 0;i<shape_len;i++)
			feature_size *= shape[i];
		std::vector<float> feature(feature_size);

		MXPredGetOutput(pred_feature, 0, feature.data(), feature_size);

		cv::Mat output = cv::Mat(feature, true).reshape(1, 1);
		cv::normalize(output, output);

		return output;
	}

private:
	PredictorHandle pred_feature;
};


vector<cv::Rect> recognition(MxNetMtcnn& mtcnn, Mxnet_extract& extract, cv::Mat& img, const cv::Mat& data)
{
	cv::Mat src(5, 2, CV_32FC1, norm_face);

	std::vector<face_box> face_info;

	//detect with mtcnn, get detect reuslt with bounding box and landmark point
	mtcnn.Detect(img, face_info);

//    cout << "i. " << face_info.size() << endl;
    vector<cv::Rect> face_boxes;
	for (int i = 0; i < face_info.size(); ++i)
	{
		NDArray::WaitAll();
	    face_box face = face_info[i];

		float v2[5][2] =
			{ { face.landmark.x[0] , face.landmark.y[0] },
			{ face.landmark.x[1] , face.landmark.y[1] },
			{ face.landmark.x[2] , face.landmark.y[2] },
			{ face.landmark.x[3] , face.landmark.y[3] },
			{ face.landmark.x[4] , face.landmark.y[4] } };
        face_boxes.push_back(cv::Rect(face.x0, face.y0, face.x1 - face.x0, face.y1 - face.y0));
		cv::Mat dst(5, 2, CV_32FC1, v2);

		//do similar transformation according normal face
		cv::Mat m = similarTransform(dst, src);
		cv::Mat aligned(112, 112, CV_32FC3);
		cv::Size size(112, 112);

		//get aligned face with transformed matrix and resize to 112*112
		cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
//        cv::Mat transfer = cv::getAffineTransform(src, dst);

//		cv::imwrite("img.png", img);
//		cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);
		cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);
//        cv::imwrite("aligned.png", aligned);
        if (CONFIG::_USE_GRAY)
            aligned = get_gray_image(aligned);

        if (CONFIG::_USE_BRIGHTNESS)
            aligned = increase_brightness(aligned);

        if (CONFIG::_USE_NORM)
            aligned = norm_img(aligned);


		//extract feature from aligned face and do classification with labels
		cv::Mat output = extract.extractFeature(aligned);
//        class_info result = classify(output, data);
        verification_info results = verify(output, data);

		//draw landmark points
		for (int j = 0; j < 5; j++)
		{
			cv::Point p(face.landmark.x[j], face.landmark.y[j]);
			cv::circle(img, p, 2, cv::Scalar(0, 0, 255), -1);
		}

		//draw bound ing box
		cv::Point pt1(face.x0, face.y0);
		cv::Point pt2(face.x1, face.y1);
		cv::rectangle(img, pt1, pt2, cv::Scalar(0, 255, 0), 2);

		cv::Point pt3(face.x0, face.y0 - 10);
        if (results.distance < -25 && results.similarity > 0.1){
            cv::putText(img, "Accept", pt3, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0));
        } else{
            cv::putText(img, "Reject", pt3, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0));
        }

        cv::Point ptd(10, 20);
        cv::putText(img, "Distance: " + to_string(results.distance), ptd, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
        cv::Point pts(10, 40);
        cv::putText(img, "Similarity: " + to_string(results.similarity), pts, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));

	}

	return face_boxes;
}


cv::Mat get_features(std::vector<face_box> face_info, Mxnet_extract& extract, cv::Mat& img)
{
	cv::Mat src(5, 2, CV_32FC1, norm_face);
	int idx = 0;
	for (int i = 0; i < face_info.size(); ++i)
	{
		NDArray::WaitAll();
//        cout << "i. " << i << endl;
	    face_box face = face_info[i];

		float v2[5][2] =
			{ { face.landmark.x[0] , face.landmark.y[0] },
			{ face.landmark.x[1] , face.landmark.y[1] },
			{ face.landmark.x[2] , face.landmark.y[2] },
			{ face.landmark.x[3] , face.landmark.y[3] },
			{ face.landmark.x[4] , face.landmark.y[4] } };

		cv::Mat dst(5, 2, CV_32FC1, v2);

		//do similar transformation according normal face
		cv::Mat m = similarTransform(dst, src);

		cv::Mat aligned(112, 112, CV_32FC3);
		cv::Size size(112, 112);

		//get aligned face with transformed matrix and resize to 112*112
		cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
		cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);

        cv::imwrite("../saved/" + std::to_string(idx) + "pure.png", aligned);

        if (CONFIG::_USE_GRAY)
            aligned = get_gray_image(aligned);

        if (CONFIG::_USE_BRIGHTNESS)
            aligned = increase_brightness(aligned);

        if (CONFIG::_USE_NORM)
            aligned = norm_img(aligned);

        cv::imwrite("../saved/" + std::to_string(idx) + ".png", aligned);
        idx++;
		//extract feature from aligned face and do classification with labels
		cv::Mat output = extract.extractFeature(aligned);

        return output;
    }

}