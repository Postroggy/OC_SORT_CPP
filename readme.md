# 介绍
本项目是C++版本的OC-SORT(OC-SORT: Observation-Centric SORT on video Multi-Object Tracking)，矩阵运算使用的库是 Eigen。  
本项目主要参考了 [OC_SORT官方Python实现](https://github.com/noahcao/OC_SORT)。  
在代码逻辑和变量命令上尽量与官方Python版本的保持一致，线性分配算法使用了开源库[Lap](https://github.com/gatagat/lap/tree/master/lap)。  
OC-SORT中改进的Kalman Filter只使用了Eigen库实现。

后续我可能会尝试发布将OCSORT与检测器接合的部署在资源有限设备上的应用。
- 参见下文安卓上跑的Demo。

# OC-Sort on Android Device
Thanks to [FeiGeChuanShu](https://github.com/FeiGeChuanShu/ncnn-android-yolov8), 修改了一下他的代码，将OC-Sort缝上去了，代码见`android-demo/`文件夹。

下载地址: [Release Apk](https://github.com/Postroggy/OC_SORT_CPP/releases/tag/v1.0.0)


## 编译环境
NCNN库一直都在发布新版本，如果你想用最新的，可以换掉，但是注意对应的NDK匹配问题。
- Android Studio
- NDK(25.2.9519653)
- Ncnn(20230223)
- Cmake(3.31.1)
- Gradle(8.7.3)

## Yolo模型
使用的是Nano和Small两个尺寸的模型。

如果想使用更大尺寸的模型，可以参考NCNN官方文档如何转换，然后放在`app\src\main\assets`目录下；再修改`strings.xml`和`yolov8ncnn.cpp`中的`modeltypes`变量即可。

# 运行速度
当前我的设备CPU是:`Ryzen R5 2500U`，编译的时候开启`-O2`优化，平均处理一帧的时间是`5.5ms`。我实现的这版本确实比ByteTrack的C++版本要慢，但是Python原版的比ByteTrack的慢特别多，代码重构成C++还是有提升的，可以在生产环境下试一试了。

# 用法
下载代码库
```bash
git clone https://github.com/Postroggy/OC_SORT_CPP.git --recursive
```
下载vcpkg二进制文件
```bash
cd OC_SORT_CPP/externals/vcpkg
./bootstrap-vcpkg.bat -useSystemBinaries # windows
./bootstrap-vcpkg.sh -useSystemBinaries # linux
```
CMake编译命令（使用 Ninja）：
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -G Ninja -S [path-to]/OC_SORT_CPP -B [path-to]/OC_SORT_CPP/cmake-build-debug
cmake --build [path-to]/OC_SORT_CPP/cmake-build-debug
```
或者使用 Visual Studio（Windows）：
```bash
cmake -G "Visual Studio 17 2022" -A x64 -S [path-to]/OC_SORT_CPP -B [path-to]/OC_SORT_CPP/build
cmake --build [path-to]/OC_SORT_CPP/build --config Release
```
使用vcpkg的manifest mode，依赖会自动下载。

`src`文件夹是对头文件中定义的函数的实现，`include`文件夹负责定义头文件。使用时将整个 OC_SORT 打包成动态链接库即可。  
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
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/externals/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
project(OC_SORT_CPP)

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) # MSVC required
file(GLOB SRC_LIST src/*.cpp)

# compile as a DLL named OCLib
add_library(OCLib SHARED ${SRC_LIST})
target_include_directories(OCLib PUBLIC include)
target_link_libraries(OCLib Eigen3::Eigen)

add_executable(test test.cpp)
target_include_directories(test PUBLIC include)
target_link_libraries(test PUBLIC Eigen3::Eigen OCLib)
```

`test.cpp`中的内容：[见文件](https://github.com/Postroggy/OC_SORT_CPP/blob/master/test.cpp)

# 测试

项目提供三个测试程序：

| 程序 | 说明 | 依赖 |
|------|------|------|
| `test_MOT` | 读取 MOT17 官方格式 TXT 文件 | 无 |
| `test` | 读取 CSV 文件夹进行测试 | 无 |
| `test_vis` | 可视化测试，显示追踪结果 | OpenCV |

## 快速测试（无需下载数据集）

### 方式1：使用 test_MOT
```bash
./test_MOT
# 自动读取 test_data/MOT17-02.txt
```

### 方式2：使用 test
```bash
./test [csv_folder] [num_frames]
# 默认: ./test test_data/MOT17-01 450
```

## 可视化测试（需要 OpenCV 和 MOT17 数据集）

如果你想运行 `test_vis` 进行可视化测试，需要启用 `visualization` feature 并准备视频文件。

### 0. 启用 visualization feature（自动安装 OpenCV）
配置 CMake 时添加 feature 参数：
```bash
# 使用 Ninja
cmake -DCMAKE_BUILD_TYPE=Debug -DVCPKG_MANIFEST_FEATURES="visualization" -G Ninja -S . -B build

# 或者使用 Visual Studio
cmake -DVCPKG_MANIFEST_FEATURES="visualization" -G "Visual Studio 17 2022" -A x64 -S . -B build
```

> **注意**: 首次启用会下载并编译 OpenCV，可能需要较长时间。

### 1. 下载 MOT17 数据集
访问 [MOTChallenge 官网](https://motchallenge.net/data/MOT17/) 下载 MOT17 Training Set。

下载后解压，目录结构如下：
```
MOT17/
└── train/
    ├── MOT17-02-DPM/
    │   ├── img1/           ← 图片序列
    │   │   ├── 000001.jpg
    │   │   ├── 000002.jpg
    │   │   └── ...
    │   ├── det/
    │   └── gt/
    ├── MOT17-02-FRCNN/     ← 也可以用这个，图片是一样的
    └── MOT17-02-SDP/       ← 也可以用这个，图片是一样的
```

> **注意**: `MOT17-02-DPM`、`MOT17-02-FRCNN`、`MOT17-02-SDP` 这三个文件夹内的图片序列完全相同，区别只是检测器不同。选择任意一个即可。

### 2. 使用 FFmpeg 将图片序列合成为视频
```bash
cd MOT17/train/MOT17-02-FRCNN
ffmpeg -framerate 30 -i img1/%06d.jpg -c:v libx264 -pix_fmt yuv420p MOT17-02.mp4
```

### 3. 运行可视化测试
```bash
./test_vis
# 输入视频路径，例如: ./MOT17-02.mp4 或 /path/to/MOT17-02.mp4
```

> **注意**: `test_vis.cpp` 中的检测数据路径是硬编码的，你可能需要根据实际情况修改第 81 行的路径。

# 代码优化
:construction:

# 公式推导
:construction:

# 关于输入输出的格式
和原版的OCSORT稍稍有不一样的地方：
## 输入格式
输入的类型：`Eigen::Matrix<double,Eigen::Dynamic,6>`  
格式：`<x1>,<y1>,<x2>,<y2>,<confidence>,<class>`

## 输出格式
输出的类型：`Eigen::Matrix<double,Eigen::Dynamic,>`  
格式：`<x1>,<y1>,<x2>,<y2>,<ID>,<class>,<confidence>`  
这么做是为了方便OCSORT与其他的目标检测器整合形成完整的目标追踪Pipeline。


