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
    cv::namedWindow("Window", cv::WINDOW_NORMAL);
/*
    std::string name = "Window";
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    //cv::Mat img = image_queue->receive();
    cv::Mat img = cv::imread("../images/sweets.jpg");
    std::cout << "imshow()\n" ;
    cv::imshow(name, img);
*/

    std::shared_ptr<MessageQueue<cv::Mat>> image_queue(new MessageQueue<cv::Mat>);
    std::shared_ptr<MessageQueue<cv::Mat>> detection_queue(new MessageQueue<cv::Mat>);


    // Create SSD MobileNet model
    SSDModel ssd_model = SSDModel(conf_threshold, nms_threshold);

    // Read image 
    Graphic input = Graphic(img_file, ssd_model.getClassNumber());
    // Set the pointer of detection queue into the Model object and Graphic object
    input.setImageQueue(image_queue);
    input.setDetectionQueue(detection_queue);

    cv::resizeWindow("Window", input.getWindowSize());

    ssd_model.setDetectionQueue(detection_queue);

    input.thread_for_read();
    ssd_model.thread_for_detection();

    //std::this_thread::sleep_for(std::chrono::seconds(6));

    std::vector<int> classIds;
    std::vector<std::string> classNames;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    cv::Mat current_image;
    //while(cv::waitKey(1) < 0)
    int duration = (int)(1000/input.getFps());


    int count = 0;
    while(cv::waitKey(duration) < 0)
    {
        std::cout << " *** frame total = " << image_queue->getTotal() << 
                    ", count = " << count << std::endl;
        if(image_queue->getTotal() > 0 && count >= image_queue->getTotal())
        {
            std::cout << "read frame finished! count = " << count << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        current_image = image_queue->receive();
        std::cout << "   size of image_queue = " << image_queue->getSize() << std::endl;


        if(count%(input.getDetectFreq()) == 0)
        {
            ssd_model.getNextDetection(classIds, classNames, confidences, boxes);
        }
        // Detect objects
        //std::vector<int> result_indices = ssd_model.detect(current_image);

        input.drawResult(current_image, classIds, classNames, confidences, boxes);
        
        cv::imshow("Window", current_image);

        ++count;
    }
    cv::waitKey(0);
    return 0;

}
