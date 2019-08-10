#include <iostream>
#include <random>
#include <iomanip>
#include <thread>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Graphic.h"



Graphic::Graphic(std::string _img_path, int class_num) : image_path(_img_path)
{
    //cv::namedWindow(kWinName);
    setClassColor(class_num);
}

void Graphic::thread_for_read()
{
    read_thread = std::thread(&Graphic::readImage, this);
}

void Graphic::readImage()
{
    cv::VideoCapture cap(this->image_path);
    float fps = (float)cap.get(cv::CAP_PROP_FPS);
    this->_fps = fps;
    this->_detect_freq = ((int)fps)/2;
    std::cout << "width = " << (int)cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "height = " << (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    //image_width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    //image_height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "fps = " << fps << std::endl;
    
    //image = cv::imread(img_path);
    
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
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        cv::Mat frame;
        cap >> frame;
        if(frame.empty())
            break;

        // put the freame into queue
        //frame = resizeImage(frame, 600);

        if(f_count%_detect_freq == 0)
        {
   
            cv::Mat frame1 = frame;
            //detect_queue->send(std::move(frame.clone()));
            detect_queue->send(std::move(frame1));
            std::cout << " --- send detect queue!\n";
            ++d_count;
        }
        image_queue->send(std::move(cv::Mat(frame)));
        // send the frame for detection once per second

        ++f_count;
    }
    image_queue->setTotal(f_count);
    detect_queue->setTotal(d_count);

    // Send an empty frame to prevent SSDModel::objectDetection from 
    //  keeping waiting in detect_queue->receive()
    detect_queue->send(std::move(cv::Mat()));
    std::cout << "msg_qaueue.total = v" << image_queue->getTotal() << std::endl;
    std::cout << "detect_queue.total = " << detect_queue->getTotal() << std::endl;
}

// Resize image to a fixed size.
// resized_w :　width(px) of resized image
cv::Mat Graphic::resizeImage(const cv::Mat &image_orig, const int resized_w=600)
{
    if(image_orig.cols > resized_w)
    {
        int resized_h = image_orig.rows * ((float)resized_w/(float)image_orig.cols);
        cv::Mat image_new;
        cv::resize(image_orig, image_new, cv::Size(resized_w, resized_h));
        return image_new;
    }
    return image_orig;
}

void Graphic::setClassColor(int class_num)
{
    //std::random_device random_device;
    //std::mt19937 random_engine(random_device());
    std::mt19937 random_engine(2019);
    std::uniform_int_distribution<int> distribution(0, 255);

    for(int i = 0; i < class_num; ++i)
    {
        cv::Scalar color = cv::Scalar(distribution(random_engine),
                                      distribution(random_engine),
                                      distribution(random_engine));
        class_color.push_back(color);
    }
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

void Graphic::drawResult(cv::Mat &image, std::vector<int> &classIds, std::vector<std::string> &classNames,
                            std::vector<float> &confidences, std::vector<cv::Rect> &boxes)
{
    // Measure time
    auto start = std::chrono::steady_clock::now();

    for(size_t i = 0; i < classIds.size(); i++)
    {
        std::cout << i << " : class = " << classNames[i] << 
                    ", conf = " << confidences[i] << std::endl;
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
    //cv::imshow(kWinName, image);
     // Caltulate time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> 
                            (std::chrono::steady_clock::now() - start);
    std::cout << "duration(draw) = " << duration.count() << std::endl;
}

Graphic::~Graphic()
{
    read_thread.join();
}