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
# opencv 不是必须的，如果需要，请在 vcpkg.json 处加上再编译
#set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv4")
#find_package(OpenCV REQUIRED)


set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
file(GLOB SRC_LIST src/*.cpp)

add_library(OCLib SHARED ${SRC_LIST})
target_include_directories(OCLib PUBLIC include)
target_link_libraries(OCLib Eigen3::Eigen)

# note： 主测试用的是 test
# add_executable(test test.cpp)
# target_link_libraries(test PUBLIC Eigen3::Eigen OCLib ${OpenCV_LIBS})

# 直接读取MOT17官方的格式转为输入数据
add_executable(test_MOT read_MOTtxt.cpp)
target_link_libraries(test_MOT PUBLIC Eigen3::Eigen OCLib)


# add_executable(L L.cpp)

# add_executable(test_MOT test_rollback.cpp)
# target_link_libraries(test_MOT PUBLIC)
# note： 测试用C++读取 csv 文件，形成 Matrix
# add_executable(read_TS read_tensor.cpp)
# target_link_libraries(read_TS PUBLIC Eigen3::Eigen)
# # note: 测试 assocaite 函数
# add_executable(test_assc test_associate.cpp)
# target_link_libraries(test_assc PUBLIC Eigen3::Eigen OCLib)
# # note: 测试 lapjv 函数
# add_executable(test_assignment test_assignment.cpp)
# target_link_libraries(test_assignment PUBLIC Eigen3::Eigen OCLib)

# note: 测试 MOT17数据集
# add_executable(test_MOT17 test_MOT17.cpp)
# target_link_libraries(test_MOT17 PUBLIC Eigen3::Eigen OCLib)

# note: 生成所有的数据
# add_executable(all_result generate_all_result.cpp )
# target_link_libraries(all_result PUBLIC Eigen3::Eigen OCLib)