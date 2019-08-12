#include <iostream>
#include <random>
#include <iomanip>
#include <thread>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Graphic.h"


// Constructor.
// Parameter: path to the image file, stored to the private attribute
//            number of classes, used to generate colors for classes
Graphic::Graphic(std::string _img_path, int class_num) : image_path(_img_path)
{
    setClassColor(class_num);

    // Open the video once, get information, then close the video
    cv::VideoCapture cap(image_path);
    _fps = (float)cap.get(cv::CAP_PROP_FPS);
    _detect_freq = ((int)_fps)/4;
    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    window_size = resizedSize(cv::Size(width, height));

    std::cout << "Image file : " << image_path << std::endl;
    std::cout << "- original width = " << width << std::endl;
    std::cout << "- original height = " << height << std::endl;
    std::cout << "- fps = " << _fps << std::endl;

    cap.release();
}

Graphic::~Graphic()
{
    read_thread.join();
}

void Graphic::thread_for_read()
{
    read_thread = std::thread(&Graphic::readImage, this);
}

cv::Size Graphic::getWindowSize()
{
    return window_size;
}

void Graphic::setImageQueue(std::shared_ptr<MessageQueue<cv::Mat>> _image_queue)
{
    image_queue = _image_queue;
}

void Graphic::setDetectionQueue(std::shared_ptr<MessageQueue<cv::Mat>> _detect_queue)
{
    detect_queue = _detect_queue;
}

float Graphic::getFps()
{
    return _fps;
}

int Graphic::getDetectFreq()
{
    return _detect_freq;
}

// Draw the result of detection on image(reference parameter)
void Graphic::drawResult(cv::Mat &image, 
                         const std::vector<int> &classIds, 
                         const std::vector<std::string> &classNames,
                         const std::vector<float> &confidences,
                         const std::vector<cv::Rect> &boxes)
{
    for(size_t i = 0; i < classIds.size(); i++)
    {
        // Box
        cv::Point p1 = cv::Point(boxes[i].x, boxes[i].y);
        cv::Point p2 = cv::Point(boxes[i].x + boxes[i].width, boxes[i].y + boxes[i].height);
        CV_Assert(classIds[i] < class_color.size());
        cv::rectangle(image, p1, p2, class_color[classIds[i]], 2);

        // Label
        std::ostringstream streamObj;
        streamObj << std::fixed << std::setprecision(2) << confidences[i]*100.0;
        std::string label = classNames[i]  + " : " + streamObj.str();

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int top = std::max(boxes[i].y, labelSize.height);
        cv::Point lp1 = cv::Point(boxes[i].x, top - labelSize.height-2);
        cv::Point lp2 = cv::Point(boxes[i].x + labelSize.width, top);
        cv::rectangle(image, lp1, lp2, class_color[classIds[i]], cv::FILLED);
        cv::putText(image, label, cv::Point(boxes[i].x, top-1), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(), 1);
    }
}


// Read frames from the image file(movie), and send it 
//  to image_queue and detect_queue
void Graphic::readImage()
{
    cv::VideoCapture cap(this->image_path);
    
    if(!cap.isOpened())
    {
        CV_Error(cv::Error::StsError, "Image file (" + image_path + ") cannot open.");
    }
    
    int f_count = 0;
    int d_count = 0;
    
    CV_Assert(this->detect_queue != nullptr);
    CV_Assert(this->image_queue != nullptr);

    while(true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        cv::Mat frame;
        cap >> frame;
        if(frame.empty())
            break;

        // Send to the detect_queue once per counts specified by _detect_freq
        if(f_count%_detect_freq == 0)
        {
            cv::Mat frame1 = frame;
            detect_queue->send(std::move(frame1));
            ++d_count;
        }
        image_queue->send(std::move(cv::Mat(frame)));

        ++f_count;
    }
    // Send the total counts to the queue
    image_queue->setTotal(f_count);
    detect_queue->setTotal(d_count);

    // Send an empty frame to prevent SSDModel::objectDetection from 
    //  keeping waiting in detect_queue->receive()
    detect_queue->send(std::move(cv::Mat()));
}

// Calculate the resized window size.
// This function returns the resized size where:
//   (width=600 or height=600) && (width <= 600 or height <= 600)
cv::Size Graphic::resizedSize(cv::Size orig)
{
    int w = 600;
    int h = orig.height * (float)w/(float)orig.width;
    if(h > 600)
    {
        int h_orig = h;
        h = 600;
        w = w * ((float)h / (float)h_orig);
    }
    return cv::Size(w, h);
}

// Assign a color to each class
void Graphic::setClassColor(int class_num)
{
    std::mt19937 random_engine(2019); // Use a fixed seed to get same colors always
    std::uniform_int_distribution<int> distribution(0, 255);

    for(int i = 0; i < class_num; ++i)
    {
        cv::Scalar color = cv::Scalar(distribution(random_engine),
                                      distribution(random_engine),
                                      distribution(random_engine));
        class_color.push_back(color);
    }
}
