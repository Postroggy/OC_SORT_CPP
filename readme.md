# 介绍
本项目是C++版本的OC-SORT(OC-SORT: Observation-Centric SORT on video Multi-Object Tracking)，矩阵运算使用的库是 Eigen。  
本项目主要参考了 (OC_SORT官方Python实现)[https://github.com/noahcao/OC_SORT]，在代码逻辑和变量命令上尽量与官方Python版本的保持一致，线性分配算法使用了开源库(Lap)[https://github.com/gatagat/lap/tree/master/lap] 的实现。  
OC-SORT中改进的Kalman Filter只使用了Eigen库实现。


# 用法
首先你需要库有：(Eigen)[https://eigen.tuxfamily.org/index.php?title=Main_Page]。

下载代码库
```bash
git clone https://github.com/Postroggy/OC_SORT_CPP.git
cd OC_SORT_CPP
```
将`src`文件夹是对头文件中定义的函数的实现，`include`文件夹负责定义头文件。使用时将整个 OC_SORT 打包成动态链接库即可。  
## 示例(使用CMake)
假设文件目录如下
```text
├───include
├───src
├───test.cpp
└───CMakeLists.txt
```

`CMakeLists.txt`内容如下：
```cmake
cmake_minimum_required(VERSION 3.21)
project(OC_SORT_CPP)

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) # MSVC required
file(GLOB SRC_LIST src/*.cpp)

# compile as a DLL named OCLib
add_library(OCLib SHARED ${SRC_LIST})
target_include_directories(OCLib PUBLIC include)
target_link_libraries(OCLib Eigen3::Eigen)

add_executable(OCSORT test.cpp)
target_link_directories(OCSORT PUBLIC include)
target_link_libraries(test PUBLIC Eigen3::Eigen OCLib)
```

`test.cpp`中的内容：(见文件)[https://github.com/Postroggy/OC_SORT_CPP/test.cpp]

# 代码优化
:construction:

# 公式推导
:construction:



