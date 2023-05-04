## CMakeLists.txt for src directory
## Find all source file in the current directory and assign them to the
## variable "SRC_LIST"
#set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) # MSVC 的特色？
#file(GLOB SRC_LIST *.cpp)
#find_package(Eigen3 REQUIRED)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
## Create a library named `OCLib` from the source files
#add_library(OCLib SHARED ${SRC_LIST})
#target_include_directories(OCLib PUBLIC ../include/)
#target_link_libraries(OCLib Eigen3::Eigen )