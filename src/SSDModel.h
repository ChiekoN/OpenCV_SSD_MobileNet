// SSDModel class
#ifndef SSDMODEL_H
#define SSDMODEL_H
#include <opencv2/dnn.hpp>
#include <queue>

#include "MessageQueue.h"


class SSDModel {
  public:

    SSDModel(float _conf_threshold, float _nms_threshold);
    ~SSDModel();

    // Functions  
    /*  static void callback(int pos, void *userdata); */
    // Getter
    int &getDetectedClassId(int index);
    std::string &getDetectedClassName(int index);
    float &getDetectedConfidence(int index);
    cv::Rect &getDetectedBox(int index);
    int getClassNumber();

    void thread_for_detection();
    void setDetectionQueue(std::shared_ptr<MessageQueue<cv::Mat>> _detect_queue);
    void getNextDetection(std::vector<int> &classIds,
                          std::vector<std::string> &classNames,
                          std::vector<float> &confidences,
                          std::vector<cv::Rect> &boxes);


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
    std::shared_ptr<MessageQueue<cv::Mat>> detect_queue;
    std::queue<std::vector<int>> queue_classIds;
    std::queue<std::vector<std::string>> queue_classNames;
    std::queue<std::vector<float>> queue_confs;
    std::queue<std::vector<cv::Rect>> queue_boxes;
    std::mutex _mutex;
    std::condition_variable _cond;
    
    //std::vector<int> indices; // indices of Non-Max Supression

    std::thread detection_thread;

    // Functions

    void readClassFile(); // Read class File
    void loadModel();

    std::vector<int> detect(const cv::Mat &image, std::vector<int> &classIds,
                                    std::vector<float> &confidences,
                                    std::vector<cv::Rect> &boxes);
    void objectDetection();
    void getResult();

 
   
};

#endif