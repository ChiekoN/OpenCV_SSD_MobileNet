# CPPND : Capstone Project
## Object Detection with SSD MobileNet model (Capstone Option 1)

## Overview
I implemented the object detection model using OpenCV. The Deep Neural Network model I employed here is SSD(Single Shot MultiBox Detector) with MobileNet. This program reads an image file, which could be a single photo or a movie, and performs object detection, then shows the image with indicators(box for the object detected, category name, and confidence(%)). This model can detect 90 categories of objects (listed in [`model/object_detection_classes_coco.txt`](model/object_detection_classes_coco.txt)).

I decided to build this project because I learned Deep Learning before and I was interested in making programs in which Deep Learning techinques are used by C++. After experiments with prototypes, I realized performing object detection through the network would be computationally expensive, which made the movie with object detection very slow. So I decided to create a thread which execute object detection in an independent time sequence from the main thread. The most challenging part was to get the result of detection and draw it on the right frame without interupting playing the movie.


## Structure

### Outline
This program firstly takes command line options specified by the user and sets them to inner variables.

It launches a thread which reads frames of the movie from the image file (if it's a photo, just one frame). In this thread, frames being read are sent to "image queue", as well as to "detection queue" once in a certain number of frames. This happens in `Graphic` class.

Then the other thread is launched in `SSDModel` class, which obtains an image data from "detection queue" and performs object detection. The result is stored in a queue inside the class.

After, in the main thread, image data in "image queue" is retrieved one by one. In that loop, once in a certain number of frames the result of the detection is updated by popping from the queue in `SSDModel`. The result of the detection is drawn on the image data. The image is shown in a window.

### Files and Classes
- `main.cpp`: Includes `main()` function.
 - Takes command line options and set parameters into inner variables.
 - Creates `image_queue` and `detection_queue`, create `SSDModel` object and `Graphic` object, and call functions in both objects which launch threads.
 - In a loop, it reads image data from the queue, get the result of detection, draw it on the image and show.


- `Graphic.h` `Graphic.cpp`: Define `Graphic` class.
 - Launch a thread which reads the image file
 - Draw the result of detection on the image
 - Store information about the image


- `SSDModel.h` `SSDModel.cpp` : Define `SSDModel` class.
 - Load the DNN model
 - Launch a thread which performs object detection
 - Store the result of detection in a queue, and retrieve it


- `MessageQueue.h` : Define `MessageQueue` class.
 - Holds a queue, and provides functions to send and receive frames using the queue.
 - Provides functions which sends and receives the total number of frames sent

## Contents
This repository contains:
- `src/` : Source files listed above
- `model/` : Files for the DNN model
 - SSD MobileNet model file : `frozen_inference_graph.pb` (download *ssd_mobilenet_v2_coco* from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md))
 - SSD MobileNet config file : `ssd_mobilenet_v2_coco_2018_03_29.pbtxt` (download from [here](https://github.com/opencv/opencv_extra/tree/master/testdata/dnn))
 - class file : `object_detection_classes_coco.txt` (download from [here](https://github.com/opencv/opencv/tree/master/samples/data/dnn))


- `images/` : Sample photos and videos to test the program
- `result/` : Examples of output images
- `CMakeLists.txt` : cmake configuration file
- `README.md` : This file

## Libraries
- OpenCV >= 4 (already installed in Udacity's workspace)
  - [Install instruction for Linux](https://docs.opencv.org/4.1.1/d7/d9f/tutorial_linux_install.html)
  - [Install istruction for Windows](https://www.learnopencv.com/install-opencv-4-on-windows/)
  - [Install instruction for Mac](https://www.learnopencv.com/install-opencv-4-on-macos/)


## Build
In the root directory (this repository), execute the command below:
1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make`

The executable(`ssd_obj_detect`) is created in the current directory(`build`).

## Run
In `build` directory, run the executable like below:

`./ssd_obj_detect [options] <image file>`

This reads an image file and show the result of object detection. `<image file>` should be a path to the file you want to input.

#### Options
 - `-c` : specifies *confidence threshold* between 0 and 1.0. If omitted, default value is 0.5. (example: `-c=0.3`)
 - `-n` : specifies the threshold used for Non-max Suppression between 0 and 1.0. If omitted, default value is 0.5. (example: `-n=0.7`)
 - `-h` `-?` `--help` `--usage`: Show usage.


#### Example with test images:

`./ssd_obj_detect -c=0.3 ../images/bunnings.JPG`

**NOTE:** In Udacity's workplace, please run the program from Visual Studio Code in DESKTOP. The output window isn't shown from workspace terminal.

## Expected Output

#### Photo
  `./ssd_obj_detect ../images/sweets.jpg`

![alt sweets](result/sweets_result.jpg)

#### Movie
  `./ssd_obj_detect ../images/schoolzone.mp3`

![alt school zone](result/schoolzone_result.gif)



## Rubrics
#### README
 - This document

#### Compiling and Testing
 - I compiled on the workspace and tested with both photos and movies and ensured that the expected images are shown.

#### Loops, Functions, I/O
- The project accepts user input and processes the input.
   - `main.cpp`(35-66): Used `cv::CommandLineParser` provided by OpenCV

#### Object Oriented Programming
- The project uses Object Oriented Programming techniques.
  - `Graphic.cpp` `SSDModel.cpp` `MessageQueue.h` : This project has three classes that have some attributes and functions.


- Classes use appropriate access specifiers for class members.
  - `Graphic.cpp` `SSDModel.cpp` `MessageQueue.h` : All class data members are specified as either `public` or `private`.

#### Memory Management
- The project makes use of references in function declarations.
 - `SSDModel.cpp`(40):
   ```
   void SSDModel::getNextDetection(std::vector<int> &classIds,
                                    std::vector<std::string> &classNames,
                                    std::vector<float> &confidences,
                                    std::vector<cv::Rect> &boxes))
   ```
 - `Graphic.cpp`(71):
   ```
   void Graphic::drawResult(cv::Mat &image,
                            const std::vector<int> &classIds,
                            const std::vector<std::string> &classNames,
                            const std::vector<float> &confidences,
                            const std::vector<cv::Rect> &boxes)
  ```

#### Concurrency
- The project uses multithreading.
 - `Graphic.cpp`(42) : creates a thread
 - `SSDModel.cpp`(30) : creates a thread


- A mutex or lock is used in the project.
 - `MessageQueue.h`(14, 24, 38, 43)


## Reference
This program is inspired by this sample implementation:

https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.cpp
