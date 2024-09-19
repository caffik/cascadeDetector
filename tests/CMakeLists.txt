include(FetchContent)
include(CTest)
find_package(OpenCV)

##############################
#   Configure paths header   #
##############################

set(TEST_FILES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_files")
configure_file(priv/paths_tests.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/paths_tests.hpp)

###############################
#   Create Output Directory   #
###############################

if (NOT EXISTS "${TEST_FILES_DIR}/output")
    FILE(MAKE_DIRECTORY "${TEST_FILES_DIR}/output")
endif ()

###################
#   Google Test   #
###################

set(BUILD_GMOCK OFF)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    FIND_PACKAGE_ARGS NAMES GTest
)

FetchContent_MakeAvailable(googletest)

#################
#   Set Tests   #
#################

include(GoogleTest)

add_executable(Tests
    test.detector.cpp
    test.detector.functions.cpp
)

target_link_libraries(Tests
    PRIVATE
        GTest::gtest_main
        det::Detector
        opencv_objdetect
        libutils::main
)

gtest_discover_tests(Tests)
