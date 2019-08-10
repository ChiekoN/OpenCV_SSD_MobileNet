#ifndef _GRAPHIC_H
#define _GRAPHIC_H

#include "SSDModel.h"
#include "MessageQueue.h"


class Graphic
{
  public:
    Graphic(std::string img_file, int class_num);
    ~Graphic();
    void drawResult(cv::Mat &image, std::vector<int> &classIds, std::vector<std::string> &classNames,
                            std::vector<float> &confidences, std::vector<cv::Rect> &boxes);
    cv::Mat getImage();
    void setImageQueue(std::shared_ptr<MessageQueue<cv::Mat>> _image_queue);
    void setDetectionQueue(std::shared_ptr<MessageQueue<cv::Mat>> _detect_queue);
    float getFps();
    int getDetectFreq();
    cv::Size getWindowSize();
    void thread_for_read();

  private:
    //cv::Mat image;
    //cv::VideoCapture cap;
    float _fps;
    int _detect_freq;
    int image_width;
    int image_height;
    cv::Size window_size;
    std::thread read_thread;
    //MessageQueue<cv::Mat> msg_queue;
    std::string image_path;

    std::shared_ptr<MessageQueue<cv::Mat>> detect_queue = nullptr;
    std::shared_ptr<MessageQueue<cv::Mat>> image_queue = nullptr;

    std::vector<cv::Scalar> class_color;
    //const std::string kWinName = "SSD MobileNet Object Detection";
    void readImage();
    void setClassColor(int class_num);
    cv::Mat resizeImage(const cv::Mat &image_orig, const int resized_w);
    cv::Size resizedSize(cv::Size orig);
};




#endif
