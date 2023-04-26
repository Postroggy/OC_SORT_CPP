#include "../include/association.h"
namespace ocsort{
    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd> speed_direction_batch(const Eigen::MatrixXd &dets,
                                                                       const Eigen::MatrixXd &tracks) {
        Eigen::VectorXd CX1 = (dets.col(0) + dets.col(2)) / 2.0;
        Eigen::VectorXd CY1 = (dets.col(1) + dets.col(3)) / 2.f;
        Eigen::MatrixXd CX2 = (tracks.col(0) + tracks.col(2)) / 2.f;
        Eigen::MatrixXd CY2 = (tracks.col(1) + tracks.col(3)) / 2.f;
        Eigen::MatrixXd dx = CX1.transpose().replicate(2, 1) - CX2.replicate(1, 2);
        Eigen::MatrixXd dy = CY1.transpose().replicate(2, 1) - CY2.replicate(1, 2);
        Eigen::MatrixXd norm = (dx.array().square() + dy.array().square()).sqrt() + 1e-6f;
        dx = dx.array() / norm.array();
        dy = dy.array() / norm.array();
        return std::make_tuple(dy,dx);
    }
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ocsort::associate(Eigen::MatrixXd detections, Eigen::MatrixXd trackers, float iou_threshold, Eigen::MatrixXd velocities, Eigen::MatrixXd previous_obs, float vdc_weight) {
        if(trackers.rows() == 0 ) return std::make_tuple(Eigen::VectorXd::Zero(0,2),Eigen::VectorXd::LinSpaced(detections.rows(), 0, detections.rows()-1),Eigen::VectorXd::Zero(0,5));

        Eigen::MatrixXd Y, X;
        auto result = speed_direction_batch(detections, previous_obs);
        Y = std::get<0>(result);
        X = std::get<1>(result);

    }
}