#ifndef OC_SORT_CPP_ASSOCIATION_H
#define OC_SORT_CPP_ASSOCIATION_H
#include "Eigen/Dense"
namespace ocsort{

    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> associate(Eigen::MatrixXd detections, Eigen::MatrixXd trackers,float iou_threshold,  Eigen::MatrixXd velocities,Eigen::MatrixXd previous_obs,float vdc_weight);
}

#endif//OC_SORT_CPP_ASSOCIATION_H
