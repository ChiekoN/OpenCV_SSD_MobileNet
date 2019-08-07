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
#include "Graphic.h"


/*
void callback(int pos, void *userdata)
{
    conf_threshold = pos * 0.01f;
}
*/

int main(int argc, char** argv)
{
    // Get command line options.

    const cv::String keys = 
        "{help h usage ? |      | print this message. }"
        "{c conf         |   .5 | Confidence threshold. }"
        "{n nms          |   .5 | Non-max suppression threshold. }"
        "{@input         |<none>| Input image or movie file. }";
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("SSD MobileNet Object Detection with C++");
    if(parser.has("help") || argc == 1)
    {
        parser.printMessage();
        return 0;
    }
    float conf_threshold = parser.get<float>("c");
    float nms_threshold = parser.get<float>("n");

    // Check if the input file has been specified properly.
    std::string img_file = parser.get<std::string>("@input");
    if(img_file == "")
    {
        std::cout << "Input file is not specified.\n";
        parser.printMessage();
        return 0;
    }
    if(!cv::utils::fs::exists(img_file))
    {
        std::cout << "Input file (" << img_file << ") not found.\n" ;
        return 0;
    }


    // Create SSD MobileNet model
    SSDModel ssd_model = SSDModel(conf_threshold, nms_threshold);

    // Read image 
    Graphic imageObj = Graphic(img_file);
    imageObj.setClassColor(ssd_model.getClassNumber());

    // Detect objects
    std::vector<int> result_indices = ssd_model.detect(imageObj.getImage());
    imageObj.drawResult(ssd_model, result_indices);

    cv::waitKey(0);
    
    return 0;
}
