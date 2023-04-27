# CMakeLists.txt for src directory
# Find all source file in the current directory and assign them to the
# variable "SRC_LIST"
file(GLOB SRC_LIST *.cpp)
find_package(Eigen3 REQUIRED)
# Create a library named `MyLibrary` from the source files
add_library(OCLib SHARED ${SRC_LIST})

target_include_directories(OCLib PUBLIC ../include)

target_link_libraries(OCLib Eigen3::Eigen)