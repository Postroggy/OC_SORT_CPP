cmake_minimum_required(VERSION 3.24)
project(OC_SORT_CPP)

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)


include_directories(src)
#add_executable(OC_SORT_CPP main.cpp  include/kalmanfilter.h include/KalmanBoxTracker.h include/OCsort.h include/association.h)
#target_link_libraries(OC_SORT_CPP Eigen3::Eigen)
#add_executable(test test.cpp include/OCsort.h include/kalmanfilter.h include/KalmanBoxTracker.h include/Utilities.h src/OCsort.cpp src/kalmanfilter.cpp src/KalmanBoxTracker.cpp include/association.h src/association.cpp)
add_subdirectory(src)
add_executable(test test.cpp)
target_link_directories(test PUBLIC include)
target_link_libraries(test Eigen3::Eigen OCLib)
