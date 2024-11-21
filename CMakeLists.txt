cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(xor-neural)

add_subdirectory(int)

#add_executable(${PROJECT_NAME})
#file(GLOB_RECURSE ${PROJECT_NAME}_SOURCES "src/*.cpp")
#target_sources(${PROJECT_NAME} PRIVATE ${${PROJECT_NAME}_SOURCES})