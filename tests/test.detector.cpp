#include <gtest/gtest.h>

#include "detector.hpp"
#include "paths_tests.hpp"

class DetectorTest : public ::testing::Test {
protected:
  det::Detector detector;
};

TEST_F(DetectorTest, ReadDirectory) {
  const std::filesystem::path directory{kImagesPath};

  EXPECT_TRUE(detector.readDirectory(directory));
}

TEST_F(DetectorTest, ReadDirectoryCorrectness) {
  const std::filesystem::path directory{kImagesPath};
  detector.readDirectory(directory);

  std::set<std::string> expected{"desert.jpg", "cat_1.jpg", "cat_2.png",
                                 "cat_3.jpg",  "cat_4.jpg", "cat_5.jpg"};

  for (auto &name : detector.getImagesPaths()) {
    EXPECT_TRUE(expected.find(name.filename().string()) != expected.end());
  }
}

TEST_F(DetectorTest, ReadDirectoryWithImageType) {
  const std::filesystem::path directory{kImagesPath};
  detector.readDirectory(directory, {".jpg"});

  std::set<std::string> expected{"desert.jpg", "cat_1.jpg", "cat_3.jpg",
                                 "cat_4.jpg", "cat_5.jpg"};

  for (auto &name : detector.getImagesPaths()) {
    EXPECT_TRUE(expected.find(name.filename().string()) != expected.end());
  }
}

TEST_F(DetectorTest, ReadXMLValidFile) {
  const std::filesystem::path xml{kHaarCascadeXML};
  EXPECT_TRUE(detector.readXML(xml));
}

TEST_F(DetectorTest, LoadImages) {
  const std::filesystem::path directory{kImagesPath};
  detector.readDirectory(directory);
  detector.loadImages(cv::IMREAD_GRAYSCALE);
  EXPECT_FALSE(detector.getImages().empty());
}

TEST_F(DetectorTest, Detect) {
  const std::filesystem::path directory{kImagesPath};
  detector.readDirectory(directory);
  detector.loadImages(cv::IMREAD_GRAYSCALE);
  detector.readXML(kHaarCascadeXML);
  detector.detect();
  EXPECT_FALSE(detector.getDetectedPositions().empty());
}

TEST_F(DetectorTest, DetectWithParameters) {
  const std::filesystem::path directory{kImagesPath};
  detector.readDirectory(directory);
  detector.loadImages(cv::IMREAD_GRAYSCALE);
  detector.readXML(kHaarCascadeXML);

  auto resize = [](cv::Mat &image) {
    cv::resize(image, image, cv::Size(100, 100));
  };

  detector.detect(1.02, 3, cv::Size(), cv::Size(), resize);
  EXPECT_FALSE(detector.getDetectedPositions().empty());
}

TEST_F(DetectorTest, OrderDetectedObjects) {
  const std::filesystem::path directory{kImagesPath};
  detector.readDirectory(directory);
  detector.loadImages(cv::IMREAD_GRAYSCALE);
  detector.readXML(kHaarCascadeXML);

  auto resize_and_blur = [](cv::Mat &image) {
    cv::resize(image, image, cv::Size(100, 100));
    cv::GaussianBlur(image, image, cv::Size(9, 9), 1, 1);
  };

  detector.detect(1.02, 3, cv::Size(), cv::Size(), resize_and_blur);
  detector.orderDetectedObjects();

  for (const auto &weights : detector.getWeights()) {
    EXPECT_TRUE(
        std::is_sorted(weights.begin(), weights.end(), std::greater<>()));
  }
  for (const auto &levels : detector.getLevels()) {
    EXPECT_TRUE(std::is_sorted(levels.begin(), levels.end(), std::greater<>()));
  }
}

class DetectorDetectWriteTest : public ::testing::Test {
protected:
  det::Detector detector;
  DetectorDetectWriteTest() {
    const std::filesystem::path directory{kImagesPath};
    detector.readDirectory(directory);
    detector.loadImages(cv::IMREAD_COLOR);
    detector.readXML(kHaarCascadeXML);
  }
};

TEST_F(DetectorDetectWriteTest, WriteDetectedObjects) {
  auto process = [](cv::Mat &image) {
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::resize(image, image, cv::Size(100, 100));
    cv::GaussianBlur(image, image, cv::Size(9, 9), 1, 1);
  };
  detector.detect(1.01, 3, cv::Size(), cv::Size(), process);
  const std::filesystem::path output_path{
      std::string(kDirOutput) + "/" +
      ::testing::UnitTest::GetInstance()->current_test_info()->name()};
  if (!std::filesystem::exists(output_path)) {
    std::filesystem::create_directory(output_path);
  }
  auto mark_detected_objects =
      [](cv::Mat &image, const std::vector<cv::Rect> &detected_objects) {
        for (const auto &object : detected_objects) {
          cv::rectangle(image, object, cv::Scalar(0, 255, 0), 2);
        }
      };
  EXPECT_TRUE(
      detector.writeDetectedObjects(output_path, mark_detected_objects));
}

TEST_F(DetectorDetectWriteTest, WriteDetectedObjectsWithFilter) {
  auto process = [](cv::Mat &image) {
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cv::resize(image, image, cv::Size(100, 100));
    cv::GaussianBlur(image, image, cv::Size(9, 9), 1, 1);
  };
  detector.detect(1.01, 3, cv::Size(), cv::Size(), process);
  detector.orderDetectedObjects();
  detector.filter(20, 0.4);
  const std::filesystem::path output_path{
      std::string(kDirOutput) + "/" +
      ::testing::UnitTest::GetInstance()->current_test_info()->name()};
  if (!std::filesystem::exists(output_path)) {
    std::filesystem::create_directory(output_path);
  }
  auto mark_detected_objects =
      [](cv::Mat &image, const std::vector<cv::Rect> &detected_objects) {
        for (const auto &object : detected_objects) {
          cv::rectangle(image, object, cv::Scalar(0, 255, 0), 2);
        }
      };
  EXPECT_TRUE(
      detector.writeDetectedObjects(output_path, mark_detected_objects));
}

TEST_F(DetectorDetectWriteTest, WriteDetectedObjectsBasic) {
  det::processImages(detector.getImages().begin(), detector.getImages().end(),
    [](cv::Mat& mat) {
      cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
      cv::resize(mat, mat, cv::Size(100, 100));
    } );

  detector.detect(1.01, 3, cv::Size(), cv::Size());
  const std::filesystem::path output_path{
      std::string(kDirOutput) + "/" +
      ::testing::UnitTest::GetInstance()->current_test_info()->name()};
  if (!std::filesystem::exists(output_path)) {
    std::filesystem::create_directory(output_path);
  }
  EXPECT_TRUE(
      detector.writeDetectedObjects(output_path));

}
