###############
#   OPTIONS   #
###############

option(ENABLE_TESTING "Build all tests." ON)
option(ENABLE_EXAMPLES "Build all examples." OFF)

###############
#   PROJECT   #
###############

cmake_minimum_required(VERSION 3.28)
project(CascadeDetector VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###########################
#   INCLUDE DIRECTORIES   #
###########################

add_subdirectory(scr)

###############
#   TESTING   #
###############

if(${ENABLE_TESTING})
    enable_testing()
    add_subdirectory(tests)
endif ()

################
#   EXAMPLES   #
################

if(${ENABLE_EXAMPLES})
    add_subdirectory(examples)
endif ()