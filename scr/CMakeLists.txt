###################
#   Find OpenCV   #
###################

find_package(OpenCV REQUIRED)

#######################
#   BS::thread_pool   #
#######################

set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM.cmake")

if(NOT(EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
        NAME BS_thread_pool
        GITHUB_REPOSITORY bshoshany/thread-pool
        VERSION 4.1.0
)

add_library(BS_thread_pool INTERFACE)

target_include_directories(BS_thread_pool
    INTERFACE
        ${BS_thread_pool_SOURCE_DIR}/include
)

################
#   libutils   #
################

include(FetchContent)

FetchContent_Declare(
        libutils
        GIT_REPOSITORY  https://github.com/caffik/libutils.git
        GIT_TAG         v1.1.2
)

set(ENABLE_TESTING OFF)
set(ENABLE_DOCS OFF)

FetchContent_MakeAvailable(libutils)

###########################
#   Configure libraries   #
###########################

set(SCR_DIR ${PROJECT_SOURCE_DIR}/scr)

set(SCR
    ${SCR_DIR}/detector.cpp
    ${SCR_DIR}/detector.functions.cpp
)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

set(INCLUDE
    ${INCLUDE_DIR}/detector.hpp
    ${INCLUDE_DIR}/detector.functions.hpp
)

add_library(detector SHARED ${SCR} ${INCLUDE})
add_library(det::Detector ALIAS detector)

# "${OpenCV_LIBS}"

target_link_libraries(detector
    PUBLIC BS_thread_pool opencv_core opencv_imgcodecs opencv_imgproc opencv_objdetect
    PUBLIC libutils::main
)

target_include_directories(detector
    PUBLIC
        ${INCLUDE_DIR}
)
