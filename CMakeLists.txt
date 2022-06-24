cmake_minimum_required(VERSION 3.12)

project(ocr VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
endif()

include_directories("${PROJECT_SOURCE_DIR}/onnxruntime/include")
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} onnx_mobilenetv3.cpp)

find_package(OpenCV REQUIRED)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_LIBS}
    "${PROJECT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.so"
)