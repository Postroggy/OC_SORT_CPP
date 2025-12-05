# OC-SORT C++ Implementation

<p align="center">
  <a href="./readme.md">🇨🇳 中文</a> | <b>🇬🇧 English</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue?style=flat-square&logo=cplusplus" alt="C++17">
  <img src="https://img.shields.io/badge/Eigen-3.4-green?style=flat-square" alt="Eigen3">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Android-lightgrey?style=flat-square" alt="Platform">
  <img src="https://img.shields.io/github/license/Postroggy/OC_SORT_CPP?style=flat-square" alt="License">
</p>

---

## 📖 Introduction

This project is a **C++ implementation** of OC-SORT (Observation-Centric SORT on video Multi-Object Tracking), using the **Eigen** library for matrix operations.

This implementation is based on the [Official OC-SORT Python Implementation](https://github.com/noahcao/OC_SORT). The code logic and variable naming are kept consistent with the official Python version as much as possible. The linear assignment algorithm uses the open-source library [Lap](https://github.com/gatagat/lap/tree/master/lap).

The improved Kalman Filter in OC-SORT is implemented purely with Eigen.

---

## 📱 OC-SORT on Android

Thanks to [FeiGeChuanShu](https://github.com/FeiGeChuanShu/ncnn-android-yolov8), I modified their code and integrated OC-SORT on top of it. See the `android-demo/` folder for the source code.

**📥 Download**: [Release APK](https://github.com/Postroggy/OC_SORT_CPP/releases/tag/v1.0.0)

### Build Environment
| Component | Version |
|-----------|---------|
| Android Studio | Latest |
| NDK | 25.2.9519653 |
| NCNN | 20230223 |
| CMake | 3.31.1 |
| Gradle | 8.7.3 |

> **Note**: NCNN releases new versions frequently. You can use a newer version, but pay attention to NDK compatibility.

### YOLO Models
Nano and Small models are used. For larger models, refer to the NCNN official documentation for conversion, then place them in `app\src\main\assets` and modify `modeltypes` in `strings.xml` and `yolov8ncnn.cpp`.

---

## ⚡ Performance

On my device with **Ryzen R5 2500U** CPU, compiled with `-O2` optimization, the average processing time per frame is **5.5ms**.

This C++ version is slightly slower than ByteTrack's C++ implementation, but the Python original is much slower than ByteTrack. Refactoring to C++ still provides significant improvements and is suitable for production environments.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Postroggy/OC_SORT_CPP.git --recursive
```

### 2. Bootstrap vcpkg
```bash
cd OC_SORT_CPP/externals/vcpkg

# Windows
./bootstrap-vcpkg.bat -useSystemBinaries

# Linux
./bootstrap-vcpkg.sh -useSystemBinaries
```

### 3. Build with CMake

**Using Ninja:**
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -G Ninja -S . -B build
cmake --build build
```

**Using Visual Studio (Windows):**
```bash
cmake -G "Visual Studio 17 2022" -A x64 -S . -B build
cmake --build build --config Release
```

> Dependencies are automatically downloaded via vcpkg manifest mode.

---

## 📁 Project Structure

```
OC_SORT_CPP/
├── include/          # Header files
├── src/              # Implementation files
├── test_data/        # Test data
├── android-demo/     # Android demo app
├── test.cpp          # CSV-based test
├── read_MOTtxt.cpp   # MOT17 format test
└── test_vis.cpp      # Visualization test (requires OpenCV)
```

---

## 🧪 Testing

The project provides three test programs:

| Program | Description | Dependencies |
|---------|-------------|--------------|
| `test_MOT` | Read MOT17 official format TXT file | None |
| `test` | Read CSV folder for testing | None |
| `test_vis` | Visualization test with tracking results | OpenCV |

### Quick Test (No Dataset Required)

**Using test_MOT:**
```bash
./test_MOT
# Automatically reads test_data/MOT17-02.txt
```

**Using test:**
```bash
./test [csv_folder] [num_frames]
# Default: ./test test_data/MOT17-01 450
```

### Visualization Test (Requires OpenCV and MOT17 Dataset)

#### 1. Enable visualization feature
```bash
# Using Ninja
cmake -DCMAKE_BUILD_TYPE=Debug -DVCPKG_MANIFEST_FEATURES="visualization" -G Ninja -S . -B build

# Using Visual Studio
cmake -DVCPKG_MANIFEST_FEATURES="visualization" -G "Visual Studio 17 2022" -A x64 -S . -B build
```

> **Note**: First-time enable will download and compile OpenCV, which may take a while.

#### 2. Download MOT17 Dataset
Download MOT17 Training Set from [MOTChallenge](https://motchallenge.net/data/MOT17/).

#### 3. Convert Images to Video
```bash
cd MOT17/train/MOT17-02-FRCNN
ffmpeg -framerate 30 -i img1/%06d.jpg -c:v libx264 -pix_fmt yuv420p MOT17-02.mp4
```

#### 4. Run Visualization Test
```bash
./test_vis
# Enter video path, e.g.: ./MOT17-02.mp4
```

---

## 📐 Input/Output Format

Slightly different from the original OC-SORT:

### Input Format
- **Type**: `Eigen::Matrix<double, Eigen::Dynamic, 6>`
- **Format**: `<x1>, <y1>, <x2>, <y2>, <confidence>, <class>`

### Output Format
- **Type**: `Eigen::Matrix<double, Eigen::Dynamic, 7>`
- **Format**: `<x1>, <y1>, <x2>, <y2>, <ID>, <class>, <confidence>`

This design makes it easier to integrate OC-SORT with other object detectors to form a complete tracking pipeline.

---

## 📄 CMake Example

```cmake
cmake_minimum_required(VERSION 3.21)
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/externals/vcpkg/scripts/buildsystems/vcpkg.cmake")
project(OC_SORT_CPP)

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
file(GLOB SRC_LIST src/*.cpp)

# Compile as DLL named OCLib
add_library(OCLib SHARED ${SRC_LIST})
target_include_directories(OCLib PUBLIC include)
target_link_libraries(OCLib Eigen3::Eigen)

add_executable(test test.cpp)
target_include_directories(test PUBLIC include)
target_link_libraries(test PUBLIC Eigen3::Eigen OCLib)
```

---

## 🛠️ Code Optimization
🚧 Work in Progress

## 📚 Formula Derivation
🚧 Work in Progress

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OC-SORT Official Implementation](https://github.com/noahcao/OC_SORT)
- [Lap Linear Assignment](https://github.com/gatagat/lap)
- [NCNN Android YOLOv8](https://github.com/FeiGeChuanShu/ncnn-android-yolov8)
