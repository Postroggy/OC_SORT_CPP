cmake_minimum_required(VERSION 3.24)
project(OC_SORT_CPP)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include)
include_directories(src)
#add_executable(OC_SORT_CPP main.cpp  include/kalmanfilter.h include/KalmanBoxTracker.h include/OCsort.h include/association.h)
#target_link_libraries(OC_SORT_CPP Eigen3::Eigen)
add_executable(test test.cpp)
target_link_libraries(test Eigen3::Eigen)