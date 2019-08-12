#include <iostream>
#include <string>
#include <fstream>
#include <mutex>
#include <condition_variable>
#include <thread>

#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "MessageQueue.h"
#include "SSDModel.h"


// Set input thresholds, read class list file and load the SSD MobileNet model
SSDModel::SSDModel(float _conf_threshold=0.5, float _nms_threshold=0.5) :
                    conf_threshold(_conf_threshold), nms_threshold(_nms_threshold)
{
    readClassFile();
    loadModel();
}

SSDModel::~SSDModel()
{
    detection_thread.join();
}

void SSDModel::thread_for_detection()
{
    detection_thread = std::thread(&SSDModel::objectDetection, this);
}

void SSDModel::setDetectionQueue(std::shared_ptr<MessageQueue<cv::Mat>> _detect_queue)
{
    detect_queue = _detect_queue;
}

// Get the result of detection from queues and set it to
//  reference parameters
void SSDModel::getNextDetection(std::vector<int> &classIds,
                                std::vector<std::string> &classNames,
                                std::vector<float> &confidences,
                                std::vector<cv::Rect> &boxes)
{
    /* set next queue into parameters */
    std::unique_lock<std::mutex> ulock(_mutex);
    _cond.wait(ulock, [this]{ return !queue_classIds.empty(); });
    classIds = std::move(queue_classIds.front());
    queue_classIds.pop();
    classNames = std::move(queue_classNames.front());
    queue_classNames.pop();
    confidences = std::move(queue_confs.front());
    queue_confs.pop();
    boxes = std::move(queue_boxes.front());
    queue_boxes.pop();
}

// Return the number of classes
int SSDModel::getClassNumber()
{
    return classes.size();
}


// Get images from detect_queue, perform detection,
//  and store the result in queues.
void SSDModel::objectDetection()
{
    int count = 0;
    while(true)
    {
        cv::Mat current_image = detect_queue->receive();

        if(detect_queue->getTotal() > 0 && count >= detect_queue->getTotal())
        {
            break;
        }

        std::vector<int> classIds, classIds_out;
        std::vector<float> confidences, confidences_out;
        std::vector<cv::Rect> boxes, boxes_out;
        std::vector<std::string> classNames_out;
        
        // Input the image to the model and get the result
        std::vector<int> indices = detect(current_image, classIds, confidences, boxes);
        for(int index : indices)
        {
            classIds_out.push_back(classIds[index]);
            confidences_out.push_back(confidences[index]);
            boxes_out.push_back(boxes[index]);
            classNames_out.push_back(classes[classIds[index]]);
        }
        // lock and push the result to queues
        std::lock_guard<std::mutex> ulock(_mutex);
        queue_classIds.push(std::move(classIds_out));
        queue_confs.push(std::move(confidences_out));
        queue_boxes.push(std::move(boxes_out));
        queue_classNames.push(std::move(classNames_out));
        _cond.notify_one();

        ++count;
    }   
}

// Perform object detection to a image being input.
// The output from the network is set to reference parameters(classIds, confidences, boxes).
// Return: indices which is used to pick up the final detection to show
//         from vectors listed above.  
std::vector<int> SSDModel::detect(const cv::Mat &image,
                                    std::vector<int> &classIds,
                                    std::vector<float> &confidences,
                                    std::vector<cv::Rect> &boxes)
{
    // Make a blob of (n, c, h, w)
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, sz, cv::Scalar(), swapRB, false);
    // Input the blob to the network
    net.setInput(blob, "", scale, mean);

    // Get the output blob from the network by feed forward
    std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
    std::vector<cv::Mat> outs;
    net.forward(outs, outNames);

    // The shape of output blob is 1x1xNx7, where N is a number of detections and 
    // 7 is a vector of each detection: 
    //  [batchId, classId, confidence, left, top, right, bottom]
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

                classIds.push_back(classId - 1); // classID=0 is background, and we have to start
                                                    // the index from 1 as 0 to get a corresponding
                                                    // class name from the class list.
                confidences.push_back(confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }            
        }
    }
    std::vector<int> indices;
    // Non-Max Supression               
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    return indices;
}

// Read Class file and store the class list to classes  .
void SSDModel::readClassFile()
{
    // Open and read class file
    std::ifstream ifs(class_file.c_str());
    if(!ifs.is_open())
        CV_Error(cv::Error::StsError, "Class File (" + class_file + ") not found.");

    std::string line;
    while(std::getline(ifs, line))
    {
        classes.push_back(line);
    }
}

// Load DNN model and store it to the private attribute.
void SSDModel::loadModel()
{
    net = cv::dnn::readNetFromTensorflow(model_file, config_file);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Check the output layer type
    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::string outLayerType = net.getLayer(outLayers[0])->type;
    if(outLayerType != "DetectionOutput")
       CV_Error(cv::Error::StsNotImplemented, "Unexpected output layer type: " + outLayerType);     

}

