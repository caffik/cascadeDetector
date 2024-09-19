#include "detector.hpp"
#include "detector.functions.hpp"
#include <libutils/algorithm.hpp>

namespace det {

std::pair<double, double> getScaleFactor(const cv::Size &size_from,
                                         const cv::Size &size_to) {
  if (!size_from.area() || !size_to.area()) {
    return std::make_pair(1.0, 1.0);
  }
  return std::make_pair(static_cast<double>(size_to.width) /
                            static_cast<double>(size_from.width),
                        static_cast<double>(size_to.height) /
                            static_cast<double>(size_from.height));
}

void rescaleRectangleInPlace(cv::Rect &rectangle, const double width_scale,
                           const double height_scale) {
  rectangle.x =
      static_cast<int>(static_cast<double>(rectangle.x) * width_scale);
  rectangle.y =
      static_cast<int>(static_cast<double>(rectangle.y) * height_scale);
  rectangle.width =
      static_cast<int>(static_cast<double>(rectangle.width) * width_scale);
  rectangle.height =
      static_cast<int>(static_cast<double>(rectangle.height) * height_scale);
}

cv::Rect rescaleRectangle(const cv::Rect &rectangle, const double width_scale,
                          const double height_scale) {
  cv::Rect scaled_rectangle{rectangle};
  rescaleRectangleInPlace(scaled_rectangle, width_scale, height_scale);
  return scaled_rectangle;
}

bool isSliceable(const cv::Size &size, const cv::Rect &rectangle) {
  if (rectangle.x + rectangle.width <= size.width &&
      rectangle.y + rectangle.height <= size.height) {
    return true;
  }
  return false;
}

std::pair<bool, std::size_t>
findBestDetectionIndex(const std::vector<double> &weights,
                       const std::vector<int> &levels, const double weight,
                       const unsigned int level) {
  const auto arg_max{utils::argmax_conditional(
      weights.begin(), weights.end(), levels.begin(),
      [level](const unsigned int img_level) { return img_level == level; })};
  if (arg_max.first && weights[arg_max.second] >= weight) {
    return arg_max;
  }
  return std::make_pair(false, arg_max.second);
}

std::vector<cv::Rect>::iterator
filterDetectedObjects(std::vector<cv::Rect>::iterator first_detected,
                      std::vector<cv::Rect>::iterator last_detected,
                      std::vector<int>::iterator first_level,
                      std::vector<double>::iterator first_weight,
                      const unsigned int level, const double weight) {
  for (; first_detected != last_detected;
       ++first_detected, ++first_level, ++first_weight) {
    if (*first_level < level || *first_weight < weight) {
      break;
    }
  }
  return first_detected;
}

} // namespace det