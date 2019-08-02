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

// Thresholds
float conf_threshold = .5;
float nms_threshold = .5;

void callback(int pos, void *userdata)
{
    conf_threshold = pos * 0.01f;
}

int main() {

    // Set file paths
    std::string class_file = "./model/object_detection_classes_coco.txt";
    std::string model_file = "./model/frozen_inference_graph.pb";
    std::string config_file = "./model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt";
    // std::string img_file = "./images/IMG_3083.JPG";
    // std::string img_file = "./images/IMG_3820.JPG";
    std::string img_file = "./images/IMG_3475.JPG";

    // parameters
    float scale = 1.f;
    cv::Scalar mean = cv::Scalar(0, 0, 0);
    cv::Size sz = cv::Size(300, 300);
    bool swapRB = true;
    

    std::vector<std::string> classes;

    // Open and read class file
    std::ifstream ifs(class_file.c_str());
    if(!ifs.is_open())
        CV_Error(cv::Error::StsError, "Class File (" + class_file + ") not found.");

    std::string line;
    while(std::getline(ifs, line))
    {
        classes.push_back(line);
    }
   

    // Load a model
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model_file, config_file);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    // Get output layer names
    std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();

    // Create a window.
    static const std::string kWinName = "Deep Learning object detection in OpenCV";
    // Make a window
    cv::namedWindow(kWinName, cv::WINDOW_AUTOSIZE);



    // Set confidence threshold
    int init_conf = (int)(conf_threshold * 100);
    cv::createTrackbar("confidence threshold[%]", kWinName, &init_conf, 99, callback);
    
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

    // Make a blob of (n, c, h, w)
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, sz, cv::Scalar(), swapRB, false);
    // Input the blob to the network
    net.setInput(blob, "", scale, mean);

    // Get the output blob from the network by feed forward
    std::vector<cv::Mat> outs;
    net.forward(outs, outNames);

    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::cout << "outLayerType : " << outLayerType << std::endl;
    std::cout << "outs.size() : " << outs.size() << std::endl;
    std::cout << "outs[0].dims : " << outs[0].dims << std::endl; 
    std::cout << "outs[0].rows : " << outs[0].total() << std::endl;

    // Vectors to keep the result of detection
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    if(outLayerType == "DetectionOutput")
    {
        // The shape of output blob is 1x1xNx7, where N is a number of detections and 
        // 7 is a vector of each detection for [batchId, classId, confidence, left, top, right, bottom]
        for(size_t k = 0; k < outs.size(); k++)
        {
            float *data = (float *)outs[k].data;
            for(size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i+2];
                int classId = (int)data[i+1];
                if(confidence > conf_threshold)
                {
                    float left = data[i+3] * image.cols;
                    float top = data[i+4] * image.rows;
                    float right = data[i+5] * image.cols;
                    float bottom = data[i+6] * image.rows;

                    // Add 1 because cv::Rect() defines the boundary as left and top are inclusive,
                    //  and as right and bottom are exclusive?
                    float width = right - left + 1; 
                    float height = bottom - top + 1;
                    
                    std::cout << "(l,t,r,b) = " << left << ", " << top << ", " << right << ", " << bottom << std::endl;

                    classIds.push_back(classId - 1); // classID=0 is background, and we have to start
                                                     // the index from 1 as 0 to get a corresponding
                                                     // class name from the class list.
                    confidences.push_back(confidence);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
                
                
            }
        }
    }
    else
    {
        CV_Error(cv::Error::StsNotImplemented, "Unexpected output layer type: " + outLayerType);
    }

    // Non-Max Supression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    for(size_t i = 0; i < indices.size(); i++)
    {
        int indx = indices[i];
        int classId = classIds[indx];
        float conf = confidences[indx];
        std::cout << i << " : class = " << classes[classId] << ", conf = " << conf << std::endl;
    
        cv::Point p1 = cv::Point(boxes[indx].x, boxes[indx].y);
        cv::Point p2 = cv::Point(boxes[indx].x + boxes[indx].width, boxes[indx].y + boxes[indx].height);
        cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 1);
    }
    
    cv::imshow(kWinName, image);
    //std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    cv::waitKey(0);
    
    return 0;
}
