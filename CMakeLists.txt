cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(cv_proj)

find_package(OpenCV REQUIRED)

add_executable (img_class src/img_class.cpp)
target_link_libraries (img_class ${OpenCV_LIBS})
