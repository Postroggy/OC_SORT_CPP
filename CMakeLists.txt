cmake_minimum_required(VERSION 3.21)
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/externals/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
project(OC_SORT_CPP VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4") # 开启警告
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2") # 开启编译优化
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} /O2") # 开启编译优化
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_RELEASE} /O2") # 开启编译优化

find_package(Eigen3 REQUIRED)
# opencv 不是必须的，如果需要运行test.cpp，请在 vcpkg.json 处加上opencv再编译
#find_package(OpenCV REQUIRED)


set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
file(GLOB SRC_LIST src/*.cpp)

add_library(OCLib SHARED ${SRC_LIST})
target_include_directories(OCLib PUBLIC include)
target_link_libraries(OCLib Eigen3::Eigen)

# 直接读取MOT17官方的格式转为输入数据，文件在: test_data/MOT17-02.txt 处。
add_executable(test_MOT read_MOTtxt.cpp)
target_link_libraries(test_MOT PUBLIC Eigen3::Eigen OCLib)

# note： 主测试用的是 test , 使用ocsort追踪，并且绘制到屏幕上。
add_executable(test test.cpp)
target_link_libraries(test PUBLIC Eigen3::Eigen OCLib ${OpenCV_LIBS})
