cmake_minimum_required(VERSION 3.2)
ADD_DEFINITIONS(-std=c++11)
OPTION(LINUX "Option description" OFF)
IF(LINUX)
    ADD_DEFINITIONS(-DLINUX)
ENDIF(LINUX)

PROJECT(mxnet-insightface-cpp)

FIND_PACKAGE(OpenCV REQUIRED)

MESSAGE(STATUS "    OpenCV library status:")
MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(
                     include/include
                     include/include/cpp-package
)

AUX_SOURCE_DIRECTORY(src DIR_SRCS)

LINK_DIRECTORIES(./lib)


set (dlib_DIR /media/vy/DATA/projects/face/insightface_cpp/dlib/build)
find_package(dlib REQUIRED )
find_package(BLAS REQUIRED )
find_package(LAPACK REQUIRED )
include_directories(./lib)

ADD_EXECUTABLE(mxnet-insightface-cpp ${DIR_SRCS})

TARGET_LINK_LIBRARIES(mxnet-insightface-cpp ${OpenCV_LIBS} mxnet  lapacke lapack blas dlib::dlib)


set(CMAKE_CXX_STANDARD 14)
set (OpenCV_DIR /media/vy/DATA/projects/face/insightface_cpp/opencv/opencv-4.1.1/build)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)
find_package( OpenCV REQUIRED )
find_package(dlib REQUIRED )

