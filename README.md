[![Build and Test](https://github.com/caffik/cascadeDetector/actions/workflows/build_test.yml/badge.svg)](https://github.com/caffik/cascadeDetector/actions/workflows/build_test.yml)

# CascadeDetector

CascadeDetector is a C++ library for detecting objects in images using Haar cascades. It leverages OpenCV for image
processing and detection, and provides a simple interface for reading directory, loading images, detecting objects,
and writing the results to disk.

## Features

### Detector

- Read images from a directory
- Load Haar cascade XML files
- Apply custom processing functions to images
- Detect objects in images
- Write processed images with detected objects to disk

### Utilities functions

Key functionalities include:

- Setting and getting the number of threads.
- Calculating scale factors between sizes.
- Rescaling rectangles.
- Examining if a rectangle can fit within an image.
- Finding the best detection index based on weights and levels.
- Reading and processing images.
- Filtering detected objects based on level and weight thresholds.
- Writing images to disk.

## Requirements

- C++17 or later
- OpenCV 4.x
- CMake 3.28 or later

## Installation

### Using CMake

To add CascadeDetector to your project, you can use CMake's `add_subdirectory` function. First, clone the repository:

```sh
git clone https://github.com/yourusername/CascadeDetector.git
```

Then, add the following lines to your `CMakeLists.txt`:

```cmake
# Add the CascadeDetector library
add_subdirectory(CascadeDetector)

# Link your target with CascadeDetector
target_link_libraries(your_target PRIVATE CascadeDetector)
```

Library can be also added with `FetchContent`:

```cmake
    include(FetchContent)
    FetchContent_Declare(
        CascadeDetector
        GIT_REPOSITORY
        GIT_TAG v1.0.0
    )
    FetchContent_MakeAvailable(CascadeDetector)
```

## Project Structure

- `include/`: Contains the public headers for the library.
- `src/`: Contains the implementation of the library.
- `examples/`: Contains example applications demonstrating how to use the library.
- `tests/`: Contains unit tests for the library.

## Usage

### Basic Example

Here is a basic example of how to use CascadeDetector to read images from a directory, detect objects, and write the
results to disk:

```cpp
#include <detector.hpp>

int main() {
    const std::filesystem::path directory{"path/to/images"};
    const std::filesystem::path xml_file{"path/to/haarcascade.xml"};
    const std::filesystem::path output_path{"path/to/output"};
    det::setNumThreads(12);  // Set number of threads for parallel processing
    cv::setNumThreads(2);    // Set number of threads for OpenCV
    det::Detector detector;
    detector.readDirectory(directory);
    detector.readXML(xml_file);
    detector.loadImages(cv::IMREAD_COLOR);

    auto process = [](cv::Mat &image) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        cv::resize(image, image, cv::Size(100, 100));
        cv::GaussianBlur(image, image, cv::Size(9, 9), 1, 1);
    };

    detector.detect(1.01, 3, cv::Size(), cv::Size(), process);
    detector.writeDetectedObjects(output_path);  // Write images on which objects have been detected

    return 0;
}
```

### Advanced Example

This example demonstrates how to filter detected objects and mark them on the images before saving:

```cpp
#include <detector.hpp>

int main() {
    const std::filesystem::path directory{"path/to/images"};
    const std::filesystem::path xml_file{"path/to/haarcascade.xml"};
    const std::filesystem::path output_path{"path/to/output"};
    det::setNumThreads(12);  // Set number of threads for parallel processing
    cv::setNumThreads(2);    // Set number of threads for OpenCV
    
    det::Detector detector;
    detector.readDirectory(directory);
    detector.readXML(xml_file);
    detector.loadImages(cv::IMREAD_COLOR);

    auto process = [](cv::Mat &image) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        cv::resize(image, image, cv::Size(100, 100));
        cv::GaussianBlur(image, image, cv::Size(9, 9), 1, 1);
    };

    detector.detect(1.01, 3, cv::Size(), cv::Size(), process);
    detector.orderDetectedObjects();  // Order detecions by chance of being correct
    detector.filter(20, 0.4);         // Removes detections which level is below 20 and weight is below 0.4

    auto mark_detected_objects = [](cv::Mat &image, const std::vector<cv::Rect> &detected_objects) {
        for (const auto &object : detected_objects) {
            cv::rectangle(image, object, cv::Scalar(0, 255, 0), 2);
        }
    };

    detector.writeDetectedObjects(output_path, mark_detected_objects);

    return 0;
}
```

## Running Tests

To run the tests, you need to have Google Test installed. You can then build and run the tests using CMake:

```sh
mkdir build
cd build
cmake ..
make
ctest
```

This will build the project and run all the unit tests in the `tests/` directory.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
