cmake_minimum_required(VERSION 3.7)

set(CMAKE_CXX_STANDARD 17)
message("CMAKE_CXX_STANDARD = ${CMAKE_CXX_STANDARD}")

project(SSD_objectdetection)
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++17 -pthread")
set(CMAKE_CXX_FLAGS "-pthread")

find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Find all executables
file(GLOB project_SRCS src/*.cpp) #src/*.h

# Add project executable
add_executable(ssd_obj_detect ${project_SRCS})
target_link_libraries(ssd_obj_detect ${OpenCV_LIBRARIES})

