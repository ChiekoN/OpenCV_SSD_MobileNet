#ifndef _GRAPHIC_H
#define _GRAPHIC_H

#include "SSDModel.h"

class Graphic {
  public:
    Graphic(std::string img_file);
    void drawResult(SSDModel &ssd_model, std::vector<int> &indices);
    void setClassColor(int class_num);
    cv::Mat &getImage();

  private:
    cv::Mat image;
    std::vector<cv::Scalar> class_color;
    const std::string kWinName = "SSD MobileNet Object Detection";
};




#endif
