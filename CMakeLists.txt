cmake_minimum_required(VERSION 3.15)
project(SmartMouse)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(Tesseract REQUIRED)

if(WIN32)
    # Windows specific
    set(PLATFORM_LIBS user32)
elseif(APPLE)
    # macOS specific
    find_library(APP_SERVICES ApplicationServices)
    set(PLATFORM_LIBS ${APP_SERVICES})
else()
    # Linux specific
    find_package(X11 REQUIRED)
    set(PLATFORM_LIBS X11 Xtst)
endif()

add_executable(smart_mouse smart_mouse.cpp)

target_include_directories(smart_mouse PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${Tesseract_INCLUDE_DIRS}
)

target_link_libraries(smart_mouse 
    ${OpenCV_LIBS}
    ${Tesseract_LIBRARIES}
    ${PLATFORM_LIBS}
)
