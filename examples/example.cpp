#include "paths_examples.hpp"
#include <detector.hpp>

int main() {
  const std::filesystem::path output_path{std::string(kExamplesDir) +
                                          "/output"};

  const std::filesystem::path cats_images_output_path{output_path / "cats"};

  if (!std::filesystem::exists(output_path)) {
    std::filesystem::create_directory(output_path);
  }

  if (!std::filesystem::exists(cats_images_output_path)) {
    std::filesystem::create_directory(cats_images_output_path);
  }

  cv::setNumThreads(2);
  det::setNumThreads(std::thread::hardware_concurrency());

  const std::filesystem::path directory{kDataSetCatImagesDir};
  det::Detector detector;
  detector.readDirectory(directory, {".jpg"});

  detector.readXML(kHaarCatsFileDir);

  auto process = [](cv::Mat &image) {
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::resize(image, image, cv::Size(100, 100));
    cv::GaussianBlur(image, image, cv::Size(9, 9), 1, 1);
  };
  detector.loadImages(cv::IMREAD_COLOR);
  detector.detect(1.01, 3, cv::Size(), cv::Size(), process);
  auto mark_objects = [](cv::Mat &image, const std::vector<cv::Rect> &objects) {
    for (const auto &object : objects) {
      cv::rectangle(image, object, cv::Scalar(0, 255, 0), 2);
    }
  };
  detector.writeDetectedObjects(cats_images_output_path, mark_objects);
  return 0;
}