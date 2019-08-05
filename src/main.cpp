//
// SDD object detection using OpenCV
//   - Using SSD MobileNet v2 COCO data with TensorFlow
//
// configration file (.pbtxt) downloaded from below:
// https://github.com/opencv/opencv_extra/tree/master/testdata/dnn
//
// SDD MobileNet model file (.pb) downloaded from below:
// https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
//
// Sample source:
// https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/core/utils/filesystem.hpp>

#include "SSDModel.h"


/*
void callback(int pos, void *userdata)
{
    conf_threshold = pos * 0.01f;
}
*/

int main() {

    // std::string img_file = "./images/IMG_3083.JPG";
    // std::string img_file = "./images/IMG_3820.JPG";
    // std::string img_file = "../images/IMG_3475.JPG";
    std::string img_file = "../images/sweets.jpg";

    float conf_threshold = .5;
    float nms_threshold = .5;
   
    SSDModel ssd_model = SSDModel(conf_threshold, nms_threshold);

    // Create a window.
    static const std::string kWinName = "Deep Learning object detection in OpenCV";
    // Make a window
    cv::namedWindow(kWinName, cv::WINDOW_AUTOSIZE);


    /*
    // Set confidence threshold
    int init_conf = (int)(conf_threshold * 100);
    cv::createTrackbar("confidence threshold[%]", kWinName, &init_conf, 99, SSDModel::callback);
    */

    // Open an image file
    cv::Mat image = cv::imread(img_file);
    if(image.empty())
    {
        CV_Error(cv::Error::StsError, "Image file (" + img_file + ") cannot open.");
    }

    // If image is larger than 600px in width, resize it
    const int resize_w = 600;
    if(image.cols > resize_w)
    {
        int resize_h = image.rows * ((float)resize_w/(float)image.cols);
        cv::Mat image_orig = image;
        cv::resize(image_orig, image, cv::Size(resize_w, resize_h));
    }

    
    std::vector<int> indices = ssd_model.detect(image);

    for(size_t i = 0; i < indices.size(); i++)
    {
        std::cout << i << " : class = " << ssd_model.getDetectedClassName(indices[i]) << 
                    ", conf = " << ssd_model.getDetectedConfidence(indices[i]) << std::endl;
        cv::Rect box = ssd_model.getDetectedBox(indices[i]);
        cv::Point p1 = cv::Point(box.x, box.y);
        cv::Point p2 = cv::Point(box.x + box.width, box.y + box.height);
        cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 1);
    }
    
    cv::imshow(kWinName, image);
    //std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    cv::waitKey(0);
    
    return 0;
}
