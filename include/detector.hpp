#ifndef DETECTOR_H
#define DETECTOR_H

#include <filesystem>
#include <libutils/type_traits.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "detector.functions.hpp"

namespace det {

namespace _details {} // namespace _details

/**
 * @class Detector
 * @brief A class for detecting objects in images using OpenCV.
 *
 * The Detector class provides functionalities to read images from directories,
 * process them, and detect objects using a cascade classifier. It supports
 * multithreaded image loading and processing.
 */
class Detector {
public:
  Detector() = default;
  Detector(Detector &&) = default;
  Detector &operator=(Detector &&) = default;
  ~Detector() = default;

  Detector(const Detector &) = delete;
  Detector &operator=(const Detector &) = delete;

  /**
   * @brief Reads all files from a directory.
   * @param directory The path to the directory.
   * @return True if the directory exists and files are read successfully,
   * false otherwise.
   */
  bool readDirectory(const std::filesystem::path &directory);

  /**
   * @brief Reads files of specific types from a directory.
   * @param directory The path to the directory.
   * @param image_types A vector of file extensions of the image types to read.
   * @return True if the directory exists and files are read successfully, false
   * otherwise.
   */
  bool readDirectory(const std::filesystem::path &directory,
                     const std::vector<std::string> &image_types);

  /**
   * @brief Reads an XML file containing the cascade classifier parameters.
   * @param xml_file The path to the XML file.
   * @return True if the XML file is read successfully, false otherwise.
   */
  bool readXML(const std::filesystem::path &xml_file);

  /**
   * @brief Loads images from the previously read directory paths.
   * @param flag The flag specifying the color type of the loaded image. Default
   * is cv::IMREAD_COLOR.
   */
  void loadImages(cv::ImreadModes flag = cv::IMREAD_COLOR);

  /**
   * @brief Detects objects in the loaded images using a cascade classifier.
   *
   * @remark If the number of 'funcs' is non-zero, a temporary copy of the
   * processing image is created, the functions are applied to this copy, and
   * the detection is then performed on this copy.
   *
   * @note Signature of 'unary_ops' should be void(cv::Mat &).
   *
   * @param scaleFactor The factor by which the image size is reduced at each
   * image scale.
   * @param minNeighbors The minimum number of neighbors each candidate
   * rectangle should have to retain it.
   * @param minSize Minimum possible object
   * size. Objects smaller than that are ignored.
   * @param maxSize Maximum
   * possible object size. Objects larger than that are ignored.
   * @param unary_ops The functions to be applied to temporary copy of images
   * before detection.
   */
  template <typename... UnaryOps>
  void detect(double scaleFactor = 1.1, int minNeighbors = 3,
              cv::Size minSize = cv::Size(), cv::Size maxSize = cv::Size(),
              UnaryOps &&...unary_ops) {

    detected_positions_.resize(images_.size());
    weights_.resize(images_.size());
    levels_.resize(images_.size());

    BS::thread_pool pool{getNumThreads()};

    pool.detach_blocks<std::size_t>(
        0, images_.size(),
        [this, scaleFactor, minNeighbors, &minSize, &maxSize,
         &unary_ops...](const unsigned int start, const unsigned int end) {
          cv::CascadeClassifier cc;
          cc.read(file_storage_.getFirstTopLevelNode());
          for (auto i{start}; i < end; ++i) {
            decltype(auto) img{utils::invoke_or_return(
                images_[i], std::forward<UnaryOps>(unary_ops)...)};
            cc.detectMultiScale(img, detected_positions_[i], levels_[i],
                                weights_[i], scaleFactor, minNeighbors, 0,
                                minSize, maxSize, true);
            if (!detected_positions_[i].empty()) {
              const auto &[fx, fy] =
                  getScaleFactor(img.size(), images_[i].size());
              rescaleRectanglesInPlace(detected_positions_[i].begin(),
                                       detected_positions_[i].end(), fx, fy);
            }
          }
        });
  }

  /**
   * @brief Orders the detected objects based on their levels and weights.
   *
   * It iterates over the detected objects and sorts them based on their levels
   * and weights. The sorting is done in descending order of levels and, in case
   * of a tie, in descending order of weights.
   */
  void orderDetectedObjects();

  /**
   * @brief Filters detected objects based on level and weight thresholds.
   *
   * It iterates over the detected objects and removes those that do not meet
   * the specified level and weight thresholds.
   *
   * @param level The minimum level threshold that a detected object must
   * satisfy.
   * @param weight The minimum weight threshold that a detected object must
   * satisfy.
   */
  void filter(unsigned int level, double weight);

  /**
   * @brief Writes the processed images (containing only a detected object)
   * to a specified directory.
   * @param path The path to the directory where images will be saved.
   * @return True if the images are written successfully, false otherwise.
   */
  bool
  writeDetectedObjects(const std::filesystem::path &path) const;

  /**
   * @brief Writes the processed images (containing only detected objects) to a
   * specified directory.
   *
   * It iterates over the detected objects and applies the
   * provided binary operations to the images before saving them to the
   * specified directory.
   *
   * @note The signature of 'binary_op' should be void(cv::Mat &, const
   * std::vector<cv::Rect> &).
   *
   * @tparam BinaryOp The types of the binary operations to be applied to the
   * images.
   * @param path The path to the directory where images will be saved.
   * @param binary_op The binary operations to be applied to the images before
   * saving.
   * @return True if the images are written successfully, false otherwise.
   */
  template <typename... BinaryOp,
            std::enable_if_t<(sizeof...(BinaryOp) > 0), bool> = true>
  bool writeDetectedObjects(const std::filesystem::path &path,
                                          BinaryOp &&...binary_op) const {
    if (!std::filesystem::exists(path)) {
      return false;
    }
    BS::thread_pool pool{getNumThreads()};
    pool.detach_blocks<std::size_t>(
        0, images_.size(),
        [this, &path, &binary_op...](const unsigned int start,
                                     const unsigned int stop) {
          for (auto i{start}; i < stop; ++i) {
            if (detected_positions_[i].empty()) {
              continue;
            }
            auto img{images_[i]};
            (std::invoke(binary_op, img, detected_positions_[i]), ...);
            const auto img_dir{path.string() + "/" +
                               paths_[i].filename().string()};
            cv::imwrite(img_dir, img);
          }
        });
    return true;
  }

  /**
   * @brief Gets the loaded images.
   * @return A reference to the vector of loaded images.
   */
  std::vector<cv::Mat> &getImages() { return images_; }

  /**
   * @brief Gets the detected positions in the images.
   * @return A reference to the vector of detected positions.
   */
  std::vector<std::vector<cv::Rect>> &getDetectedPositions() {
    return detected_positions_;
  }

  /**
   * @brief Gets the weights of the detected objects.
   * @return A reference to the vector of weights.
   */
  std::vector<std::vector<double>> &getWeights() { return weights_; }

  /**
   * @brief Gets the levels of the detected objects.
   * @return A reference to the vector of levels.
   */
  std::vector<std::vector<int>> &getLevels() { return levels_; }

  /**
   * @brief Gets the paths of the images to be processed.
   * @return A reference to the vector of image paths.
   */
  std::vector<std::filesystem::path> &getImagesPaths() { return paths_; }

private:
  std::vector<std::filesystem::path> paths_;

  std::vector<cv::Mat> images_;
  std::vector<std::vector<cv::Rect>> detected_positions_;
  std::vector<std::vector<double>> weights_;
  std::vector<std::vector<int>> levels_;

  cv::FileStorage file_storage_;
};

} // namespace det
#endif // DETECTOR_H
