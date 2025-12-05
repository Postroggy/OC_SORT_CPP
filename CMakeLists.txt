cmake_minimum_required(VERSION 3.21)
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/externals/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
project(OC_SORT_CPP VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Compiler optimization flags (cross-platform)
if(MSVC)
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4") # Enable warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")  # UTF-8 source and execution charset
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    # GCC/Clang
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
endif()

find_package(Eigen3 REQUIRED)

# OpenCV is optional. To build test.cpp, add opencv to vcpkg.json first.
find_package(OpenCV QUIET)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
file(GLOB SRC_LIST src/*.cpp)

# Build OC-SORT as a shared library
add_library(OCLib SHARED ${SRC_LIST})
target_include_directories(OCLib PUBLIC include)
target_link_libraries(OCLib Eigen3::Eigen)

# Executable for parsing MOT17 official format. Input file: test_data/MOT17-02.txt
add_executable(test_MOT read_MOTtxt.cpp)
target_link_libraries(test_MOT PUBLIC Eigen3::Eigen OCLib)

# Test executable using CSV files (no OpenCV required)
add_executable(test test.cpp)
target_link_libraries(test PUBLIC Eigen3::Eigen OCLib)

# Visualization test executable (requires OpenCV)
if(OpenCV_FOUND)
    add_executable(test_vis test_vis.cpp)
    target_link_libraries(test_vis PUBLIC Eigen3::Eigen OCLib ${OpenCV_LIBS})
else()
    message(STATUS "OpenCV not found, 'test_vis' target will not be built. Add opencv to vcpkg.json if needed.")
endif()
