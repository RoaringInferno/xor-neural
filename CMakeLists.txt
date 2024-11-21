cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(xor-neural)

add_subdirectory(int)

add_library(neural INTERFACE)
target_include_directories(neural INTERFACE include)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} neural)