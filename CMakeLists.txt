cmake_minimum_required(VERSION 3.21)
project(OC_SORT_CPP)

set(CMAKE_CXX_STANDARD 17)
SET(Eigen3_DIR "C:/Users/GYQ/vcpkg/installed/x86-windows/share/eigen3")
find_package(Eigen3 REQUIRED)
#add_subdirectory(src)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
file(GLOB SRC_LIST src/*.cpp)
add_library(OCLib SHARED ${SRC_LIST})
target_include_directories(OCLib PUBLIC include)
target_link_libraries(OCLib Eigen3::Eigen)
# note： 主测试用的是 test
#add_executable(test test.cpp)
#target_link_directories(test PUBLIC include)
#target_link_libraries(test PUBLIC Eigen3::Eigen OCLib)
# note： 测试用C++读取 csv 文件，形成 Matrix
#add_executable(read_TS read_tensor.cpp)
#target_link_libraries(read_TS PUBLIC Eigen3::Eigen)
## note: 测试 assocaite 函数
# add_executable(test_assc test_associate.cpp)
# target_link_libraries(test_assc PUBLIC Eigen3::Eigen OCLib)
## note: 测试 lapjv 函数
#add_executable(test_assignment test_assignment.cpp)
#target_link_libraries(test_assignment PUBLIC Eigen3::Eigen OCLib)

# note: 测试 MOT17数据集
add_executable(test_MOT17 test_MOT17.cpp)
target_link_libraries(test_MOT17 PUBLIC Eigen3::Eigen OCLib)