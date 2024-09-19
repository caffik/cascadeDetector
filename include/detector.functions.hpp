#ifndef DETECTOR_FUNCTIONS_H
#define DETECTOR_FUNCTIONS_H

#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

#include "BS_thread_pool.hpp"
#include "details/variables.hpp"

namespace det {

/**
 * Sets the number of threads to be used. By default, only the main thread is
 * used.
 *
 * @param num_threads The number of threads to set. If zero, it defaults to 1.
 */
static void setNumThreads(const unsigned int num_threads) {
  _details::num_threads = num_threads ? num_threads : 1;
}

/**
 * Gets the number of threads currently set.
 *
 * @return The number of threads.
 */
static unsigned int getNumThreads() { return _details::num_threads; }

/**
 * Calculates the scale factor between two sizes.
 *
 * @param size_from The original size.
 * @param size_to The target size.
 * @return A pair of doubles representing the width and height scale factors.
 *
 * @note If either size_from or size_to has an area of zero, the scale factor is
 * set to 1.0.
 */
std::pair<double, double> getScaleFactor(const cv::Size &size_from,
                                         const cv::Size &size_to);

/**
 * Scales a rectangle in place by the given width and height scale factors.
 *
 * @param rectangle The rectangle to be scaled.
 * @param width_scale The scale factor for the width.
 * @param height_scale The scale factor for the height.
 * @remark The functions scale the x, y, width, and height of the rectangle.
 */
void rescaleRectangleInPlace(cv::Rect &rectangle, double width_scale,
                           double height_scale);

/**
 * Rescales a range of rectangles in place by the given width and height scale
 * factors.
 *
 * @tparam InputIt The type of the input iterator.
 * @param first The beginning of the range of rectangles.
 * @param last The end of the range of rectangles.
 * @param width_scale The scale factor for the width.
 * @param height_scale The scale factor for the height.
 * @remark If both width_scale and height_scale are 1.0, the function returns
 * immediately.
 */
template <typename InputIt>
void rescaleRectanglesInPlace(InputIt first, InputIt last,
                              const double width_scale,
                              const double height_scale) {
  if (width_scale == 1.0 && height_scale == 1.0) {
    return;
  }
  std::for_each(first, last, [width_scale, height_scale](auto &rect) {
    rescaleRectangleInPlace(rect, width_scale, height_scale);
  });
}

/**
 * Rescales a rectangle by the given width and height scale factors.
 *
 * @param rectangle The rectangle to be scaled.
 * @param width_scale The scale factor for the width.
 * @param height_scale The scale factor for the height.
 * @return A new rectangle that has been scaled by the given factors.
 * @remark The functions scale the x, y, width, and height of the rectangle.
 */
cv::Rect rescaleRectangle(const cv::Rect &rectangle, double width_scale,
                          double height_scale);

/**
 * Checks if a given rectangle can fit within the dimensions of a size.
 * @param size The image in which the rectangle is to be checked.
 * @param rectangle The rectangle to be checked.
 * @return True if the rectangle can fit within the image dimensions, false
 * otherwise.
 */
bool isSliceable(const cv::Size &size, const cv::Rect &rectangle);

/**
 * Finds index that points to the best rectangle (detected object) that
 * satisfies the given level conditions.
 *
 * @param weights A vector of weights associated with rectangles.
 * @param levels A vector of levels associated with rectangles.
 * @param weight The minimum weight threshold that a rectangle must satisfy.
 * @param level The specific level that a rectangle must match.
 * @return A pair where the first element is a boolean indicating if a suitable
 * rectangle was found, and the second element is the index of the best
 * rectangle found.
 */
std::pair<bool, std::size_t>
findBestDetectionIndex(const std::vector<double> &weights,
                       const std::vector<int> &levels, double weight,
                       unsigned int level);

namespace _details {
/**
 * Reads images from a range of input paths with a specified flag.
 *
 * @tparam InputIt The type of the input iterator.
 * @tparam OutputIt The type of the output iterator.
 * @param first The beginning of the range of input paths.
 * @param last The end of the range of input paths.
 * @param d_first The beginning of the destination range where the images will
 * be stored.
 * @param flag The flag specifying the color type of the loaded image. Default
 * is cv::IMREAD_COLOR.
 */
template <typename InputIt, typename OutputIt>
void readImages(InputIt first, InputIt last, OutputIt d_first,
                const cv::ImreadModes flag = cv::IMREAD_COLOR) {
  std::transform(first, last, d_first,
                 [&flag](const auto &path) { return cv::imread(path, flag); });
}
} // namespace _details

/**
 * Reads images from a range of input paths with a specified flag using
 * getNumThreads() threads.
 *
 * @tparam InputIt The type of the input iterator.
 * @tparam OutputIt The type of the output iterator.
 * @param first The beginning of the range of input paths.
 * @param last The end of the range of input paths.
 * @param d_first The beginning of the destination range where the images will
 * be stored.
 * @param flag The flag specifying the color type of the loaded image. Default
 * is cv::IMREAD_COLOR.
 */
template <typename InputIt, typename OutputIt>
void readImages(InputIt first, InputIt last, OutputIt d_first,
                const cv::ImreadModes flag = cv::IMREAD_COLOR) {
  if (getNumThreads() == 1) {
    return _details::readImages(first, last, d_first, flag);
  }
  BS::thread_pool pool{getNumThreads()};
  const auto loops{std::distance(first, last)};

  pool.detach_blocks<int>(
      0, loops, [&d_first, &first, &flag](const int start, const int end) {
        _details::readImages(std::next(first, start), std::next(first, end),
                   std::next(d_first, start), flag);
      });
  pool.wait();
}

namespace _details {
/**
 * Processes images by applying a series of functions to each image in the
 * range.
 *
 * @tparam InputIt The type of the input iterator.
 * @tparam Funcs The types of the functions to be applied to each image.
 * @param first The beginning of the range of input images.
 * @param last The end of the range of input images.
 * @param funcs The functions to be applied to each image.
 */
template <typename InputIt, typename... Funcs>
void processImages(InputIt first, InputIt last, Funcs... funcs) {
  for (; first != last; ++first) {
    (std::invoke(funcs, *first), ...);
  }
}
} // namespace _details

/**
 * Processes images by applying a series of functions to each image in the range
 * using multiple threads if specified.
 *
 * @tparam InputIt The type of the input iterator.
 * @tparam Funcs The types of the functions to be applied to each image.
 * @param first The beginning of the range of input images.
 * @param last The end of the range of input images.
 * @param funcs The functions to be applied to each image.
 */
template <typename InputIt, typename... Funcs>
void processImages(InputIt first, InputIt last, Funcs... funcs) {
  if (getNumThreads() == 1) {
    return _details::processImages(first, last, funcs...);
  }

  BS::thread_pool pool{getNumThreads()};
  const auto loops{std::distance(first, last)};

  pool.detach_blocks<int>(
      0, loops, [&first, &funcs...](const int start, const int end) {
        _details::processImages(std::next(first, start), std::next(first, end),
                                std::forward<Funcs>(funcs)...);
      });
  pool.wait();
}

namespace _details {
/**
 * Reads and processes images from a range of input paths.
 *
 * @tparam InputIt The type of the input iterator.
 * @tparam OutputIt The type of the output iterator.
 * @tparam Funcs The types of the functions to be applied to each image.
 * @param first The beginning of the range of input paths.
 * @param last The end of the range of input paths.
 * @param d_first The beginning of the destination range where processed images
 * will be stored.
 * @param flag The flag specifying the color type of the loaded image. Default
 * is cv::IMREAD_COLOR.
 * @param funcs The functions to be applied to each image.
 */
template <typename InputIt, typename OutputIt, typename... Funcs>
void readAndProcessImages(InputIt first, InputIt last, OutputIt d_first,
                          const cv::ImreadModes flag = cv::IMREAD_COLOR,
                          Funcs... funcs) {
  for (; first != last; ++first, ++d_first) {
    *d_first = cv::imread(*first, flag);
    (std::invoke(funcs, *d_first), ...);
  }
}
} // namespace _details

/**
 * Reads and processes images from a range of input paths using getNumThreads()
 * threads.
 *
 * @tparam InputIt The type of the input iterator.
 * @tparam OutputIt The type of the output iterator.
 * @tparam Funcs The types of the functions to be applied to each image.
 * @param first The beginning of the range of input paths.
 * @param last The end of the range of input paths.
 * @param d_first The beginning of the destination range where processed images
 * will be stored.
 * @param flag The flag specifying the color type of the loaded image. Default
 * is cv::IMREAD_COLOR.
 * @param funcs The functions to be applied to each image.
 */
template <typename InputIt, typename OutputIt, typename... Funcs>
void readAndProcessImages(InputIt first, InputIt last, OutputIt d_first,
                          const cv::ImreadModes flag = cv::IMREAD_COLOR,
                          Funcs... funcs) {
  if (getNumThreads() == 1) {
    return _details::readAndProcessImages(first, last, d_first, flag, funcs...);
  }

  BS::thread_pool pool{getNumThreads()};
  const auto loops{std::distance(first, last)};

  pool.detach_blocks<int>(
      0, loops,
      [&d_first, &first, &flag, &funcs...](const int start, const int end) {
        _details::readAndProcessImages(
            std::next(first, start), std::next(first, end),
            std::next(d_first, start), flag, std::forward<Funcs>(funcs)...);
      });
  pool.wait();
}

/**
 * @brief Filters detected objects based on level and weight thresholds.
 *
 * This function iterates over a range of detected objects and filters them
 * based on the specified level and weight thresholds. It returns an
 * iterator to the first object that does not meet the criteria.
 *
 * @note It is assumed that the range of detected objects, levels, and weights
 * is sorted in descending order of levels and weights. Please see
 * Detector::orderDetectedObjects() for more information.
 *
 * @param first_detected The beginning of the range of detected objects.
 * @param last_detected The end of the range of detected objects.
 * @param first_level The beginning of the range of levels.
 * @param first_weight The beginning of the range of weights.
 * @param level The minimum level threshold that a detected object must satisfy.
 * @param weight The minimum weight threshold that a detected object must
 * satisfy.
 * @return An iterator to the first object that does not meet the criteria.
 * If such an object is not found, the function returns last_detected.
 */
std::vector<cv::Rect>::iterator
filterDetectedObjects(std::vector<cv::Rect>::iterator first_detected,
                      std::vector<cv::Rect>::iterator last_detected,
                      std::vector<int>::iterator first_level,
                      std::vector<double>::iterator first_weight,
                      const unsigned int level, const double weight);

/**
 * @brief Writes a range of images to disk.
 *
 * This function iterates over a range of images and their corresponding file
 * paths, writing each image to the specified path using `cv::imwrite`.
 *
 * @tparam InputIt1 The type of the input iterator for the images.
 * @tparam InputIt2 The type of the input iterator for the file paths.
 * @param first_images The beginning of the range of images.
 * @param last_images The end of the range of images.
 * @param first_paths The beginning of the range of file paths.
 */
template <typename InputIt1, typename InputIt2>
void writeImages(InputIt1 first_images, InputIt1 last_images,
                 InputIt2 first_paths) {
  for (; first_images != last_images; ++first_images, ++first_paths) {
    cv::imwrite(*first_paths, *first_images);
  }
}

} // namespace det

#endif
