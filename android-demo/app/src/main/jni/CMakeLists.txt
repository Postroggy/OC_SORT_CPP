cmake_minimum_required(VERSION 3.10)
project(yolov8ncnn)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Werror,-Wformat-security")

include_directories(ocsort/include)
include_directories(eigen)
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.6.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230223-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)


add_library(yolov8ncnn SHARED yolov8ncnn.cpp yolo.cpp ndkcamera.cpp
        ${CMAKE_SOURCE_DIR}/ocsort/src/lapjv.cpp
        ${CMAKE_SOURCE_DIR}/ocsort/src/KalmanFilter.cpp
        ${CMAKE_SOURCE_DIR}/ocsort/src/KalmanBoxTracker.cpp
        ${CMAKE_SOURCE_DIR}/ocsort/src/association.cpp
        ${CMAKE_SOURCE_DIR}/ocsort/src/OCsort.cpp
        ${CMAKE_SOURCE_DIR}/ocsort/src/Utilities.cpp
)
target_link_libraries(yolov8ncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
