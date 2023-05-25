cmake_minimum_required(VERSION 3.16)

project(ocr VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type: Release." FORCE)
endif()

include_directories("${PROJECT_SOURCE_DIR}/onnxruntime/include")
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} examples/onnx_mobilenetv3.cpp)

find_package(OpenCV REQUIRED)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_LIBS}
    "${PROJECT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.so"
)

target_compile_options(${PROJECT_NAME} 
    PRIVATE 
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /utf-8 /DUNICODE /D_UNICODE /DNOMINMAX /Zc:preprocessor /Zc:__cplusplus /wd5054>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wno-deprecated-enum-enum-conversion>
)