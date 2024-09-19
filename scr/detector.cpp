#include <libutils/files.hpp>
#include <libutils/iterator.hpp>

#include "detector.functions.hpp"
#include "detector.hpp"

namespace det {

bool Detector::readDirectory(const std::filesystem::path &directory) {
  if (!std::filesystem::exists(directory) &&
      std::filesystem::is_directory(directory)) {
    return false;
  }
  paths_ = utils::read_directory(directory);
  return true;
}

bool Detector::readDirectory(const std::filesystem::path &directory,
                             const std::vector<std::string> &image_types) {
  if (!std::filesystem::exists(directory)) {
    return false;
  }
  auto predicate = [&image_types](const std::filesystem::path &path) {
    return std::find(image_types.begin(), image_types.end(),
                     path.extension()) != image_types.end();
  };
  paths_ = utils::read_directory_if(directory, predicate);
  return true;
}

bool Detector::readXML(const std::filesystem::path &xml_file) {
  file_storage_.open(xml_file, cv::FileStorage::READ);
  if (!file_storage_.isOpened()) {
    return false;
  }
  return true;
}

void Detector::loadImages(const cv::ImreadModes flag) {
  images_.resize(paths_.size());
  readImages(paths_.begin(), paths_.end(), images_.begin(), flag);
}

void Detector::orderDetectedObjects() {
  BS::thread_pool pool{getNumThreads()};
  utils::MultiIterator absolut_begin{levels_.begin(), weights_.begin(),
                                     detected_positions_.begin()};

  pool.detach_blocks<std::size_t>(
      0, detected_positions_.size(),
      [absolut_begin](const unsigned int start, const unsigned int stop) {
        utils::MultiIterator begin{absolut_begin + start};
        utils::MultiIterator end{absolut_begin + stop};

        std::for_each(begin, end, [](auto &&rref) {
          utils::MultiIterator inner_begin{utils::get<0>(rref).begin(),
                                           utils::get<1>(rref).begin(),
                                           utils::get<2>(rref).begin()};
          utils::MultiIterator inner_end{utils::get<0>(rref).end(),
                                         utils::get<1>(rref).end(),
                                         utils::get<2>(rref).end()};
          std::sort(inner_begin, inner_end,
                    [](const auto &lhs, const auto &rhs) {
                      if (utils::get<0>(lhs) == utils::get<0>(rhs)) {
                        return utils::get<1>(lhs) > utils::get<1>(rhs);
                      }
                      return utils::get<0>(lhs) > utils::get<0>(rhs);
                    });
        });
      });
}

void Detector::filter(const unsigned int level, const double weight) {
  BS::thread_pool pool{getNumThreads()};

  pool.detach_blocks<std::size_t>(
      0, detected_positions_.size(),
      [this, &level, &weight](const unsigned int start,
                              const unsigned int stop) {
        for (auto i{start}; i < stop; ++i) {
          if (detected_positions_[i].empty()) {
            continue;
          }
          detected_positions_[i].erase(
              det::filterDetectedObjects(
                  detected_positions_[i].begin(), detected_positions_[i].end(),
                  levels_[i].begin(), weights_[i].begin(), level, weight),
              detected_positions_[i].end());
        }
      });
}

bool Detector::writeDetectedObjects(const std::filesystem::path &path) const {
  if (!std::filesystem::exists(path)) {
    return false;
  }
  for (auto i{0}; i < images_.size(); ++i) {
    if (detected_positions_[i].empty()) {
      continue;
    }
    const auto img_dir{path.string() + "/" + paths_[i].filename().string()};
    cv::imwrite(img_dir, images_[i]);
  }
  return true;
}

} // namespace det
