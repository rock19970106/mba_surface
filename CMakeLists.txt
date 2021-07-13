cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(mba_surface)

set(CMAKE_CXX_STANDARD 11)

#OpenCv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#Eigen
find_package(Eigen3)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})

#PCL
find_package(PCL 1.10 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(mba_surface main.cpp)
target_link_libraries(mba_surface ${PCL_LIBRARIES} ${OpenCV_LIBS})