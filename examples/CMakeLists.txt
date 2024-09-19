###################################
#   Download Cat and Dog images   #
###################################

message(STATUS "Examples: Configuring")

set(EXAMPLES_DIR "${PROJECT_SOURCE_DIR}/examples")
set(DATA_SET_DIR "${EXAMPLES_DIR}/dataSet")

if (NOT EXISTS "${DATA_SET_DIR}")
    FILE(MAKE_DIRECTORY dataSet)
endif ()

if (NOT EXISTS "${DATA_SET_DIR}/catsanddogs.zip")
    message(STATUS "Examples: Downloading 'catsanddog' data set")
    file(DOWNLOAD "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
        "${PROJECT_SOURCE_DIR}/examples/dataSet/catsanddogs.zip"
        SHOW_PROGRESS
    )
endif ()

if (NOT EXISTS "${DATA_SET_DIR}/PetImages")
    message(STATUS "Examples: Extracting 'catsanddog' data set")
    file(ARCHIVE_EXTRACT
    INPUT "${DATA_SET_DIR}/catsanddogs.zip"
    DESTINATION "${DATA_SET_DIR}"
    PATTERNS "PetImages/Cat/*"
             "PetImages/Dog/*"
    )
endif ()

###################################
#   Download Haarcascades files   #
###################################

set(HAAR_DIR "${EXAMPLES_DIR}/haarcascades")

if (NOT EXISTS "${HAAR_DIR}")
    FILE(MAKE_DIRECTORY haarcascades)
endif ()

set(HAAR_CATS_FILE_NAME "haarcascade_frontalcatface_extended.xml")
set(HAAR_DOGS_FILE_NAME "haarcascade_frontaldogface_extended.xml")

if (NOT EXISTS "${HAAR_DIR}/${HAAR_CATS_FILE_NAME}")
    message(STATUS "Examples: Downloading 'cats' Haar cascade")
    file(DOWNLOAD
            "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalcatface_extended.xml"
            "${HAAR_DIR}/${HAAR_CATS_FILE_NAME}"
    )
endif()

if (NOT EXISTS "${HAAR_DIR}/${HAAR_DOGS_FILE_NAME}")
    message(STATUS "Examples: Downloading 'dogs' Haar cascade")
    file(DOWNLOAD
            "https://raw.githubusercontent.com/kskd1804/dog_face_haar_cascade/master/cascade.xml"
            "${HAAR_DIR}/${HAAR_DOGS_FILE_NAME}"
    )
endif()

####################################
#   Configure paths_examples.hpp   #
####################################

configure_file(
    "${EXAMPLES_DIR}/private/paths_examples.hpp.in"
    "${EXAMPLES_DIR}/paths_examples.hpp"
)

###########################
#   Add executable files  #
###########################

add_executable(Example example.cpp)
target_link_libraries(
    Example
        PRIVATE det::Detector
)