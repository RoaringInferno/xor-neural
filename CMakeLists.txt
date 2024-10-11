cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)

project(xor-neural)

add_executable(xor-neural)
file(GLOB_RECURSE SOURCES "src/*.cpp")
target_sources(xor-neural PRIVATE ${SOURCES})
target_include_directories(xor-neural PRIVATE "include")

# Armadillo
add_subdirectory(lib/armadillo-14.0.2)
target_link_libraries(xor-neural PRIVATE armadillo)

# Benchmarks
add_subdirectory(benchmark)