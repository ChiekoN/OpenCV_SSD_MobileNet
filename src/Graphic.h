#ifndef _GRAPHIC_H
#define _GRAPHIC_H

#include "SSDModel.h"
#include "MessageQueue.h"


class Graphic
{
  public:
    Graphic(std::string img_file, int class_num);
    ~Graphic();
    void drawResult(cv::Mat &image, 
                    const std::vector<int> &classIds,
                    const std::vector<std::string> &classNames,
                    const std::vector<float> &confidences,
                    const std::vector<cv::Rect> &boxes);
    cv::Mat getImage();
    void setImageQueue(std::shared_ptr<MessageQueue<cv::Mat>> _image_queue);
    void setDetectionQueue(std::shared_ptr<MessageQueue<cv::Mat>> _detect_queue);
    float getFps();
    int getDetectFreq();
    cv::Size getWindowSize();
    void thread_for_read();

  private:
    // Information about the input video
    std::string image_path;
    float _fps;
    int _detect_freq;
    int image_width;
    int image_height;
    cv::Size window_size;
    // thread for reading images
    std::thread read_thread;
    // pointer for the queue to send images being read
    std::shared_ptr<MessageQueue<cv::Mat>> detect_queue = nullptr;
    std::shared_ptr<MessageQueue<cv::Mat>> image_queue = nullptr;
    // colors being assigned to classes randomly
    std::vector<cv::Scalar> class_color;

    void readImage();
    void setClassColor(int class_num);
    cv::Size resizedSize(cv::Size orig);
};

#endif
