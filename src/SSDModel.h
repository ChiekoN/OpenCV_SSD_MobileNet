// SSDModel class
#ifndef SSDMODEL_H
#define SSDMODEL_H
#include <opencv2/dnn.hpp>


class SSDModel {
  public:

    SSDModel(float _conf_threshold, float _nms_threshold);

    // Functions  
    /*  static void callback(int pos, void *userdata); */
    // Getter
    int &getDetectedClassId(int index);
    std::string &getDetectedClassName(int index);
    float &getDetectedConfidence(int index);
    cv::Rect &getDetectedBox(int index);
    int getClassNumber();

    std::vector<int> &detect(cv::Mat &image);

  private:

    float conf_threshold;
    float nms_threshold;

    // SSD MobileNet Model files 
    const std::string class_file = "../model/object_detection_classes_coco.txt";
    const std::string model_file = "../model/frozen_inference_graph.pb";
    const std::string config_file = "../model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt";;

    // Parameters for SSD MobileNet (fixed)
    const float scale = 1.f;
    const cv::Scalar mean = cv::Scalar(0, 0, 0);
    const cv::Size sz = cv::Size(300, 300);
    const bool swapRB = true;

    // Store the list of classe name
    std::vector<std::string> classes;
    // network object
    cv::dnn::Net net;
    // detected objects
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> indices; // indices of Non-Max Supression

    // Functions
    void readClassFile(); // Read class File
    void loadModel();

    
    void getResult();

 
   
};

#endif