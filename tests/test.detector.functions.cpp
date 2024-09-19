#include <detector.functions.hpp>
#include <gtest/gtest.h>
#include <libutils/files.hpp>

#include "paths_tests.hpp"

/* @brief Test for detector functions. */

/*
 * NumThreads tests.
 */

TEST(NumThreads, ReturnsCorrectNumberOfThreads) {
  const auto num_threads{det::getNumThreads()};
  EXPECT_EQ(num_threads, 1);
}

TEST(NumThreads, ReturnsOneWhenHardwareConcurrencyIsZero) {
  constexpr auto num_threads{10};
  det::setNumThreads(num_threads);
  EXPECT_EQ(num_threads, det::getNumThreads());
}

/*
 * getScaleFactor tests.
 */

TEST(GetScaleFactor, ReturnsCorrectScaleFactors) {
  const cv::Size size_from(100, 50);
  const cv::Size size_to(200, 100);
  const auto [fx, fy] = det::getScaleFactor(size_from, size_to);
  EXPECT_DOUBLE_EQ(fx, 2.0);
  EXPECT_DOUBLE_EQ(fy, 2.0);
}

TEST(GetScaleFactor, ReturnsOneWhenSizeFromIsZero) {
  const cv::Size size_from(0, 0);
  const cv::Size size_to(200, 100);
  const auto [fx, fy] = det::getScaleFactor(size_from, size_to);
  EXPECT_DOUBLE_EQ(fx, 1.0);
  EXPECT_DOUBLE_EQ(fy, 1.0);
}

TEST(GetScaleFactor, ReturnsOneWhenSizeToIsZero) {
  const cv::Size size_from(100, 50);
  const cv::Size size_to(0, 0);
  const auto [fx, fy] = det::getScaleFactor(size_from, size_to);
  EXPECT_DOUBLE_EQ(fx, 1.0);
  EXPECT_DOUBLE_EQ(fy, 1.0);
}

TEST(GetScaleFactor, ReturnsCorrectNonUniformScaleFactors) {
  const cv::Size size_from(100, 50);
  const cv::Size size_to(200, 150);
  const auto [fx, fy] = det::getScaleFactor(size_from, size_to);
  EXPECT_DOUBLE_EQ(fx, 2.0);
  EXPECT_DOUBLE_EQ(fy, 3.0);
}

/*
 * RescaleRectangleInPlace tests.
 */

TEST(RescaleRectangleInPlace, ScalesRectangleCorrectly) {
  cv::Rect rect(10, 20, 30, 40);
  det::rescaleRectangleInPlace(rect, 2.0, 3.0);
  EXPECT_EQ(rect.x, 20);
  EXPECT_EQ(rect.y, 60);
  EXPECT_EQ(rect.width, 60);
  EXPECT_EQ(rect.height, 120);
}

TEST(RescaleRectangleInPlace, ScalesRectangleWithOneScaleFactor) {
  cv::Rect rect(10, 20, 30, 40);
  det::rescaleRectangleInPlace(rect, 1.0, 1.0);
  EXPECT_EQ(rect.x, 10);
  EXPECT_EQ(rect.y, 20);
  EXPECT_EQ(rect.width, 30);
  EXPECT_EQ(rect.height, 40);
}

TEST(RescaleRectangleInPlace, ScalesRectangleWithZeroScaleFactor) {
  cv::Rect rect(10, 20, 30, 40);
  det::rescaleRectangleInPlace(rect, 0.0, 0.0);
  EXPECT_EQ(rect.x, 0);
  EXPECT_EQ(rect.y, 0);
  EXPECT_EQ(rect.width, 0);
  EXPECT_EQ(rect.height, 0);
}

TEST(RescaleRectangleInPlace, ScalesRectangleWithNegativeScaleFactor) {
  cv::Rect rect(10, 20, 30, 40);
  det::rescaleRectangleInPlace(rect, -1.0, -1.0);
  EXPECT_EQ(rect.x, -10);
  EXPECT_EQ(rect.y, -20);
  EXPECT_EQ(rect.width, -30);
  EXPECT_EQ(rect.height, -40);
}

/*
 * ScaleRectangle tests.
 */

TEST(RescaleRectangle, ScalesRectangleCorrectly) {
  const cv::Rect rect(10, 20, 30, 40);
  const auto scaled_rect{det::rescaleRectangle(rect, 2.0, 3.0)};
  EXPECT_EQ(scaled_rect.x, 20);
  EXPECT_EQ(scaled_rect.y, 60);
  EXPECT_EQ(scaled_rect.width, 60);
  EXPECT_EQ(scaled_rect.height, 120);
}

/*
 * rescaleRectanglesInPlace tests.
 */

TEST(RescaleRectanglesInPlace, ScalesRectanglesCorrectly) {
  std::vector<cv::Rect> rects = {{10, 20, 30, 40}, {50, 60, 70, 80}};
  det::rescaleRectanglesInPlace(rects.begin(), rects.end(), 2.0, 3.0);
  EXPECT_EQ(rects[0].x, 20);
  EXPECT_EQ(rects[0].y, 60);
  EXPECT_EQ(rects[0].width, 60);
  EXPECT_EQ(rects[0].height, 120);
  EXPECT_EQ(rects[1].x, 100);
  EXPECT_EQ(rects[1].y, 180);
  EXPECT_EQ(rects[1].width, 140);
  EXPECT_EQ(rects[1].height, 240);
}

TEST(RescaleRectanglesInPlace, ScalesRectangleWithOneScaleFactor) {
  std::vector<cv::Rect> rects = {{10, 20, 30, 40}, {50, 60, 70, 80}};
  det::rescaleRectanglesInPlace(rects.begin(), rects.end(), 1.0, 1.0);
  EXPECT_EQ(rects[0].x, 10);
  EXPECT_EQ(rects[0].y, 20);
  EXPECT_EQ(rects[0].width, 30);
  EXPECT_EQ(rects[0].height, 40);
  EXPECT_EQ(rects[1].x, 50);
  EXPECT_EQ(rects[1].y, 60);
  EXPECT_EQ(rects[1].width, 70);
  EXPECT_EQ(rects[1].height, 80);
}

/**
 * isSliceable tests.
 */

TEST(IsSliceable, RectangleFitsWithinImage) {
  const cv::Mat image(100, 100, CV_8UC3);
  const cv::Rect rectangle(10, 10, 50, 50);
  EXPECT_TRUE(det::isSliceable(image.size(), rectangle));
}

TEST(IsSliceable, RectangleExceedsImageWidth) {
  const cv::Mat image(100, 100, CV_8UC3);
  const cv::Rect rectangle(10, 10, 100, 50);
  EXPECT_FALSE(det::isSliceable(image.size(), rectangle));
}

TEST(IsSliceable, RectangleExceedsImageHeight) {
  const cv::Mat image(100, 100, CV_8UC3);
  const cv::Rect rectangle(10, 10, 50, 100);
  EXPECT_FALSE(det::isSliceable(image.size(), rectangle));
}

TEST(IsSliceable, RectangleExceedsImageWidthAndHeight) {
  const cv::Mat image(100, 100, CV_8UC3);
  const cv::Rect rectangle(10, 10, 100, 100);
  EXPECT_FALSE(det::isSliceable(image.size(), rectangle));
}

TEST(IsSliceable, RectangleFitsExactlyWithinImage) {
  const cv::Mat image(100, 100, CV_8UC3);
  const cv::Rect rectangle(0, 0, 100, 100);
  EXPECT_TRUE(det::isSliceable(image.size(), rectangle));
}

/**
 * BestConditionalRectangle tests.
 */

TEST(BestConditionalRectangle, FindsBestRectangleWithSatisfyingLevel) {
  const std::vector weights{1.0, 2.0, 3.0, 4.0, 5.0};
  const std::vector levels{1, 2, 3, 4, 5};
  const auto result{det::findBestDetectionIndex(weights, levels, 3.0, 3)};
  EXPECT_EQ(result, std::make_pair(true, static_cast<std::size_t>(2)));
}

TEST(BestConditionalRectangle, NoRectangleSatisfiesLevel) {
  const std::vector weights{1.0, 2.0, 3.0, 4.0, 5.0};
  const std::vector levels{1, 2, 3, 4, 5};
  const auto result{det::findBestDetectionIndex(weights, levels, 3.0, 6)};
  EXPECT_EQ(result, std::make_pair(false, static_cast<std::size_t>(5)));
}

TEST(BestConditionalRectangle, WeightThresholdNotMet) {
  const std::vector weights{1.0, 2.0, 3.0, 4.0, 5.0};
  const std::vector levels{1, 2, 3, 4, 5};
  const auto result{det::findBestDetectionIndex(weights, levels, 6.0, 3)};
  EXPECT_EQ(result, std::make_pair(false, static_cast<std::size_t>(2)));
}

TEST(BestConditionalRectangle, EmptyWeightsAndLevels) {
  const std::vector<double> weights = {};
  const std::vector<int> levels = {};
  const auto result{det::findBestDetectionIndex(weights, levels, 3.0, 3)};
  EXPECT_EQ(result, std::make_pair(false, static_cast<std::size_t>(0)));
}

TEST(BestConditionalRectangle, MultipleRectanglesSatisfyLevel) {
  const std::vector weights{1.0, 2.0, 3.0, 4.0, 5.0};
  const std::vector levels{3, 3, 3, 3, 3};
  const auto result{det::findBestDetectionIndex(weights, levels, 3.0, 3)};
  EXPECT_EQ(result, std::make_pair(true, static_cast<std::size_t>(4)));
}

/**
 * readImages tests.
 */

TEST(ReadImages, ReadsImagesCorrectly) {
  const std::vector paths{std::string(kImagesPath) + "/cat_1.jpg",
                          std::string(kImagesPath) + "/desert.jpg"};
  std::vector<cv::Mat> images(2);
  det::readImages(paths.begin(), paths.end(), images.begin());
  EXPECT_FALSE(images[0].empty());
  EXPECT_FALSE(images[1].empty());
}

TEST(ReadImages, ReadsImagesWithFlag) {
  const std::vector paths{std::string(kImagesPath) + "/cat.jpg",
                          std::string(kImagesPath) + "/desert.jpg"};
  std::vector<cv::Mat> images(2);
  det::readImages(paths.begin(), paths.end(), images.begin(),
                  cv::IMREAD_GRAYSCALE);
  EXPECT_EQ(images[0].channels(), 1);
  EXPECT_EQ(images[1].channels(), 1);
}

TEST(ReadImages, ReadsImagesWithFlagThreads) {
  const std::vector paths{std::string(kImagesPath) + "/cat.jpg",
                          std::string(kImagesPath) + "/desert.jpg"};
  std::vector<cv::Mat> images(2);
  det::readImages(paths.begin(), paths.end(), images.begin(),
                  cv::IMREAD_GRAYSCALE);
  EXPECT_EQ(images[0].channels(), 1);
  EXPECT_EQ(images[1].channels(), 1);
}

TEST(ReadImages, EmptyPaths) {
  const std::vector<std::string> paths = {};
  std::vector<cv::Mat> images(0);
  det::readImages(paths.begin(), paths.end(), images.begin());
  EXPECT_TRUE(images.empty());
}

TEST(ReadImages, InvalidPaths) {
  const std::vector<std::string> paths{"invalid1.jpg", "invalid2.jpg"};
  std::vector<cv::Mat> images(2);
  det::readImages(paths.begin(), paths.end(), images.begin());
  EXPECT_TRUE(images[0].empty());
  EXPECT_TRUE(images[1].empty());
}

TEST(ReadImages, MixedValidAndInvalidPaths) {
  const std::vector<std::string> paths{std::string(kImagesPath) + "/cat_1.jpg",
                                       "invalid.jpg"};
  std::vector<cv::Mat> images(2);
  det::readImages(paths.begin(), paths.end(), images.begin());
  EXPECT_FALSE(images[0].empty());
  EXPECT_TRUE(images[1].empty());
}

TEST(ReadImages, ReadsImagesCorrectlyWithThreads) {
  det::setNumThreads(2);
  const std::vector paths{std::string(kImagesPath) + "/cat_1.jpg",
                          std::string(kImagesPath) + "/desert.jpg"};
  std::vector<cv::Mat> images(2);
  det::readImages(paths.begin(), paths.end(), images.begin());
  EXPECT_FALSE(images[0].empty());
  EXPECT_FALSE(images[1].empty());
}

/**
 * processImages tests.
 */

TEST(ProcessImages, ProcessesImagesCorrectly) {
  const std::vector paths{std::string(kImagesPath) + "/cat_1.jpg",
                          std::string(kImagesPath) + "/desert.jpg"};

  std::vector<cv::Mat> images{paths.size()};
  det::readImages(paths.begin(), paths.end(), images.begin(),
                  cv::IMREAD_UNCHANGED);

  auto dummyFunc = [](cv::Mat &img) {
    cv::resize(img, img, cv::Size(100, 100));
  };

  det::processImages(images.begin(), images.end(), dummyFunc);

  for (const auto &img : images) {
    EXPECT_EQ(img.size(), cv::Size(100, 100));
  }
}

TEST(ProcessImages, ProcessesImagesCorrectlyWithThreads) {
  const std::vector paths{std::string(kImagesPath) + "/cat_1.jpg",
                          std::string(kImagesPath) + "/desert.jpg"};

  std::vector<cv::Mat> images{paths.size()};
  det::setNumThreads(2);
  det::readImages(paths.begin(), paths.end(), images.begin(),
                  cv::IMREAD_UNCHANGED);

  auto dummyFunc = [](cv::Mat &img) {
    cv::resize(img, img, cv::Size(100, 100));
  };

  det::processImages(images.begin(), images.end(), dummyFunc);

  for (const auto &img : images) {
    EXPECT_EQ(img.size(), cv::Size(100, 100));
  }
}

/*
 * ReadAndProcessImages tests.
 */

TEST(ReadAndProcessImages, ProcessesImagesCorrectly) {
  auto imagePaths{utils::read_directory(kImagesPath)};
  std::vector<cv::Mat> processedImages(imagePaths.size());
  auto dummyFunc = [](cv::Mat &img) {
    cv::resize(img, img, cv::Size(100, 100));
  };

  det::readAndProcessImages(imagePaths.begin(), imagePaths.end(),
                            processedImages.begin(), cv::IMREAD_COLOR,
                            dummyFunc);

  for (const auto &img : processedImages) {
    EXPECT_EQ(img.size(), cv::Size(100, 100));
  }
}

TEST(ReadAndProcessImages, HandlesEmptyInputRange) {
  std::vector<std::string> imagePaths;
  std::vector<cv::Mat> processedImages;

  det::readAndProcessImages(imagePaths.begin(), imagePaths.end(),
                            processedImages.begin());

  EXPECT_TRUE(processedImages.empty());
}

TEST(ReadAndProcessImages, HandlesNonExistentFiles) {
  std::vector<std::string> imagePaths = {"nonexistent1.jpg",
                                         "nonexistent2.jpg"};
  std::vector<cv::Mat> processedImages(2);

  det::readAndProcessImages(imagePaths.begin(), imagePaths.end(),
                            processedImages.begin());

  for (const auto &img : processedImages) {
    EXPECT_TRUE(img.empty());
  }
}

TEST(ReadAndProcessImages, ProcessesImagesCorrectlyWithThreads) {
  auto imagePaths{utils::read_directory(kImagesPath)};
  std::vector<cv::Mat> processedImages(imagePaths.size());
  auto dummyFunc = [](cv::Mat &img) {
    cv::resize(img, img, cv::Size(100, 100));
  };

  det::setNumThreads(2);
  det::readAndProcessImages(imagePaths.begin(), imagePaths.end(),
                            processedImages.begin(), cv::IMREAD_COLOR,
                            dummyFunc);

  for (const auto &img : processedImages) {
    EXPECT_EQ(img.size(), cv::Size(100, 100));
  }
}

/*
 * FilterDetectedObjects tests.
 */

TEST(FilterDetectedObjects, LevelAndWeightThresholdsBothMet) {
  std::vector<cv::Rect> detected_objects{{10, 10, 10, 10}};
  std::vector levels{20};
  std::vector weights{1.3};

  auto first_incorrect{det::filterDetectedObjects(
      detected_objects.begin(), detected_objects.end(), levels.begin(),
      weights.begin(), 20, 0.6)};

  EXPECT_EQ(std::distance(first_incorrect, detected_objects.end()), 0);
}

TEST(FilterDetectedObjects, LevelAndWeightThresholdsBothNotMet) {
  std::vector<cv::Rect> detected_objects{{10, 10, 10, 10}};
  std::vector levels{19};
  std::vector weights{0.3};

  auto first_incorrect{det::filterDetectedObjects(
      detected_objects.begin(), detected_objects.end(), levels.begin(),
      weights.begin(), 20, 0.6)};

  EXPECT_EQ(std::distance(first_incorrect, detected_objects.end()), 1);
}

TEST(FilterDetectedObjects, LevelThresholdNotMetWeightThresholdMet) {
  std::vector<cv::Rect> detected_objects{{10, 10, 10, 10}};
  std::vector levels{19};
  std::vector weights{1.4};

  auto first_incorrect{det::filterDetectedObjects(
      detected_objects.begin(), detected_objects.end(), levels.begin(),
      weights.begin(), 20, 0.6)};

  EXPECT_EQ(std::distance(first_incorrect, detected_objects.end()), 1);
}

TEST(FilterDetectedObjects, LevelThresholdMetWeightThresholdNotMet) {
  std::vector<cv::Rect> detected_objects{{10, 10, 10, 10}};
  std::vector levels{20};
  std::vector weights{0.2};

  auto first_incorrect{det::filterDetectedObjects(
      detected_objects.begin(), detected_objects.end(), levels.begin(),
      weights.begin(), 20, 0.6)};

  EXPECT_EQ(std::distance(first_incorrect, detected_objects.end()), 1);
}

/*
 * WriteImages tests.
 */

TEST(WriteImages, WritesImagesCorrectly) {
  const std::vector paths{std::string(kImagesPath) + "/cat_1.jpg",
                          std::string(kImagesPath) + "/desert.jpg"};
  std::vector<cv::Mat> images{paths.size()};
  det::readImages(paths.begin(), paths.end(), images.begin(),
                  cv::IMREAD_UNCHANGED);
  const auto test_name{
    ::testing::UnitTest::GetInstance()->current_test_info()->name()};
  std::filesystem::path output_path{std::string(kDirOutput) + "/" + test_name};
  if (!std::filesystem::exists(output_path)) {
    std::filesystem::create_directory(output_path);
  }
  const std::vector output_paths{
    output_path.string() + "/cat_1.jpg",
    output_path.string() + "/desert.jpg"};
  det::writeImages(images.begin(), images.end(), output_paths.begin());

  for (const auto &path : output_paths) {
    EXPECT_TRUE(std::filesystem::exists(path));
  }
}
