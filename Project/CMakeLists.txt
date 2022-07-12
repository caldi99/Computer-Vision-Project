cmake_minimum_required(VERSION 2.8)

project(projectGroup05)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

set(SOURCES source/main.cpp source/Utils.cpp source/Detector.cpp source/Segmentator.cpp include/Utils.h include/Detector.h include/Segmentator.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
