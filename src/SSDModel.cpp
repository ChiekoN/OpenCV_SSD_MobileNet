#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "SSDModel.h"

// Constructor
SSDModel::SSDModel(float _conf_threshold=0.5, float _nms_threshold=0.5) :
                    conf_threshold(_conf_threshold), nms_threshold(_nms_threshold)
{
    readClassFile();
    loadModel();
}

// Public
std::vector<int> &SSDModel::detect(cv::Mat &image)
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
                
                std::cout << "(l,t,r,b) = " << left << ", " << top << ", " << right << ", " << bottom << std::endl;

                classIds.push_back(classId - 1); // classID=0 is background, and we have to start
                                                    // the index from 1 as 0 to get a corresponding
                                                    // class name from the class list.
                confidences.push_back(confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }            
        }
    }
    // Non-Max Supression               
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    return indices;
}

/*
static void callback(int pos, void *userdata)
{
    conf_threshold = pos * 0.01f;
}
*/

std::string &SSDModel::getDetectedClassName(int index)
{
    return classes[classIds[index]];
}
float &SSDModel::getDetectedConfidence(int index)
{
    return confidences[index];
}
cv::Rect &SSDModel::getDetectedBox(int index)
{
    return boxes[index];
}

// Private
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