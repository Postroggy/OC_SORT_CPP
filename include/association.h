#ifndef OC_SORT_CPP_ASSOCIATION_H
#define OC_SORT_CPP_ASSOCIATION_H
#include "Eigen/Dense"
#include <algorithm>
#include "vector"
#include "lapjv.h"
#define pi 3.1415926

namespace ocsort {
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> speed_direction_batch(const Eigen::MatrixXd &dets,
                                                                       const Eigen::MatrixXd &tracks);
    Eigen::MatrixXd iou_batch(const Eigen::MatrixXd &bboxes1, const Eigen::MatrixXd &bboxes2);
    std::tuple<std::vector<Eigen::Matrix<int, 1, 2>>, std::vector<int>, std::vector<int>> associate(Eigen::MatrixXd detections, Eigen::MatrixXd trackers, float iou_threshold, Eigen::MatrixXd velocities, Eigen::MatrixXd previous_obs_, float vdc_weight);
}// namespace ocsort

#endif//OC_SORT_CPP_ASSOCIATION_H
