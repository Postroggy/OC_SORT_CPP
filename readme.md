# OC-SORT C++ 实现

<p align="center">
  <b>🇨🇳 中文</b> | <a href="./README_EN.md">🇬🇧 English</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue?style=flat-square&logo=cplusplus" alt="C++17">
  <img src="https://img.shields.io/badge/Eigen-3.4-green?style=flat-square" alt="Eigen3">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Android-lightgrey?style=flat-square" alt="Platform">
  <img src="https://img.shields.io/github/license/Postroggy/OC_SORT_CPP?style=flat-square" alt="License">
</p>

---

## 📖 介绍

本项目是 **C++ 版本**的 OC-SORT（Observation-Centric SORT on video Multi-Object Tracking），矩阵运算使用 **Eigen** 库。

本项目主要参考了 [OC_SORT 官方 Python 实现](https://github.com/noahcao/OC_SORT)。在代码逻辑和变量命名上尽量与官方 Python 版本保持一致，线性分配算法使用了开源库 [Lap](https://github.com/gatagat/lap/tree/master/lap)。

OC-SORT 中改进的 Kalman Filter 只使用了 Eigen 库实现。

---

## 📱 Android 演示

感谢 [FeiGeChuanShu](https://github.com/FeiGeChuanShu/ncnn-android-yolov8)，我修改了他的代码并将 OC-SORT 集成进去。源代码见 `android-demo/` 文件夹。

**📥 下载地址**: [Release APK](https://github.com/Postroggy/OC_SORT_CPP/releases/tag/v1.0.0)

### 编译环境
| 组件 | 版本 |
|------|------|
| Android Studio | 最新版 |
| NDK | 25.2.9519653 |
| NCNN | 20230223 |
| CMake | 3.31.1 |
| Gradle | 8.7.3 |

> **注意**: NCNN 库一直在发布新版本，如果你想用最新的，可以换掉，但是注意对应的 NDK 匹配问题。

### YOLO 模型
使用的是 Nano 和 Small 两个尺寸的模型。如果想使用更大尺寸的模型，可以参考 NCNN 官方文档如何转换，然后放在 `app\src\main\assets` 目录下，再修改 `strings.xml` 和 `yolov8ncnn.cpp` 中的 `modeltypes` 变量即可。

---

## ⚡ 运行速度

当前我的设备 CPU 是 **Ryzen R5 2500U**，编译时开启 `-O2` 优化，平均处理一帧的时间是 **5.5ms**。

这个 C++ 版本确实比 ByteTrack 的 C++ 版本要慢，但是 Python 原版比 ByteTrack 慢很多。将代码重构成 C++ 还是有提升的，可以在生产环境下试一试了。

---

## 🚀 快速开始

### 1. 下载代码库
```bash
git clone https://github.com/Postroggy/OC_SORT_CPP.git --recursive
```

### 2. 下载 vcpkg 二进制文件
```bash
cd OC_SORT_CPP/externals/vcpkg

# Windows
./bootstrap-vcpkg.bat -useSystemBinaries

# Linux
./bootstrap-vcpkg.sh -useSystemBinaries
```

### 3. CMake 编译

**使用 Ninja:**
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -G Ninja -S . -B build
cmake --build build
```

**使用 Visual Studio (Windows):**
```bash
cmake -G "Visual Studio 17 2022" -A x64 -S . -B build
cmake --build build --config Release
```

> 使用 vcpkg 的 manifest mode，依赖会自动下载。

---

## 📁 项目结构

```
OC_SORT_CPP/
├── include/          # 头文件
├── src/              # 实现文件
├── test_data/        # 测试数据
├── android-demo/     # Android 演示应用
├── test.cpp          # CSV 格式测试
├── read_MOTtxt.cpp   # MOT17 格式测试
└── test_vis.cpp      # 可视化测试（需要 OpenCV）
```

---

## 🧪 测试

项目提供三个测试程序：

| 程序 | 说明 | 依赖 |
|------|------|------|
| `test_MOT` | 读取 MOT17 官方格式 TXT 文件 | 无 |
| `test` | 读取 CSV 文件夹进行测试 | 无 |
| `test_vis` | 可视化测试，显示追踪结果 | OpenCV |

### 快速测试（无需下载数据集）

**方式1：使用 test_MOT**
```bash
./test_MOT
# 自动读取 test_data/MOT17-02.txt
```

**方式2：使用 test**
```bash
./test [csv_folder] [num_frames]
# 默认: ./test test_data/MOT17-01 450
```

### 可视化测试（需要 OpenCV 和 MOT17 数据集）

#### 1. 启用 visualization feature
```bash
# 使用 Ninja
cmake -DCMAKE_BUILD_TYPE=Debug -DVCPKG_MANIFEST_FEATURES="visualization" -G Ninja -S . -B build

# 使用 Visual Studio
cmake -DVCPKG_MANIFEST_FEATURES="visualization" -G "Visual Studio 17 2022" -A x64 -S . -B build
```

> **注意**: 首次启用会下载并编译 OpenCV，可能需要较长时间。

#### 2. 下载 MOT17 数据集
访问 [MOTChallenge 官网](https://motchallenge.net/data/MOT17/) 下载 MOT17 Training Set。

下载后解压，目录结构如下：
```
MOT17/
└── train/
    ├── MOT17-02-DPM/
    │   ├── img1/           ← 图片序列
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   ├── det/
    │   └── gt/
    ├── MOT17-02-FRCNN/     ← 也可以用这个，图片是一样的
    └── MOT17-02-SDP/       ← 也可以用这个，图片是一样的
```

> **注意**: `MOT17-02-DPM`、`MOT17-02-FRCNN`、`MOT17-02-SDP` 这三个文件夹内的图片序列完全相同，区别只是检测器不同。选择任意一个即可。

#### 3. 使用 FFmpeg 将图片序列合成为视频
```bash
cd MOT17/train/MOT17-02-FRCNN
ffmpeg -framerate 30 -i img1/%06d.jpg -c:v libx264 -pix_fmt yuv420p MOT17-02.mp4
```

#### 4. 运行可视化测试
```bash
./test_vis
# 输入视频路径，例如: ./MOT17-02.mp4 或 /path/to/MOT17-02.mp4
```

> **注意**: `test_vis.cpp` 中的检测数据路径是硬编码的，你可能需要根据实际情况修改第 81 行的路径。

---

## 📐 输入输出格式

和原版的 OC-SORT 稍有不同：

### 输入格式
- **类型**: `Eigen::Matrix<double, Eigen::Dynamic, 6>`
- **格式**: `<x1>, <y1>, <x2>, <y2>, <confidence>, <class>`

### 输出格式
- **类型**: `Eigen::Matrix<double, Eigen::Dynamic, 7>`
- **格式**: `<x1>, <y1>, <x2>, <y2>, <ID>, <class>, <confidence>`

这么做是为了方便 OC-SORT 与其他的目标检测器整合，形成完整的目标追踪 Pipeline。

---

## 📄 CMake 示例

```cmake
cmake_minimum_required(VERSION 3.21)
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/externals/vcpkg/scripts/buildsystems/vcpkg.cmake")
project(OC_SORT_CPP)

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
file(GLOB SRC_LIST src/*.cpp)

# 编译成名为 OCLib 的动态链接库
add_library(OCLib SHARED ${SRC_LIST})
target_include_directories(OCLib PUBLIC include)
target_link_libraries(OCLib Eigen3::Eigen)

add_executable(test test.cpp)
target_include_directories(test PUBLIC include)
target_link_libraries(test PUBLIC Eigen3::Eigen OCLib)
```

---

## 🛠️ 代码优化
🚧 施工中

## 📚 公式推导
🚧 施工中

---

## 📜 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [OC-SORT 官方实现](https://github.com/noahcao/OC_SORT)
- [Lap 线性分配](https://github.com/gatagat/lap)
- [NCNN Android YOLOv8](https://github.com/FeiGeChuanShu/ncnn-android-yolov8)
