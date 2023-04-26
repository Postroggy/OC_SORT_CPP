#include "../include/association.h"
namespace ocsort {
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> speed_direction_batch(const Eigen::MatrixXd &dets,
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
        return std::make_tuple(dy, dx);
    }
    Eigen::MatrixXd iou_batch(const Eigen::MatrixXd &bboxes1, const Eigen::MatrixXd &bboxes2) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> a = bboxes1.col(0);// bboxes1[..., 0] (2,1)
        Eigen::Matrix<double, 1, Eigen::Dynamic> b = bboxes2.col(0);// bboxes2[..., 0] (2,1) 需要转成(1,2)
        Eigen::MatrixXd xx1 = (a.replicate(1, a.rows())).cwiseMax(b.replicate(b.cols(), 1));
        a = bboxes1.col(1);
        b = bboxes2.col(1);
        Eigen::MatrixXd yy1 = (a.replicate(1, a.rows())).cwiseMax(b.replicate(b.cols(), 1));
        a = bboxes1.col(2);
        b = bboxes2.col(2);
        Eigen::MatrixXd xx2 = (a.replicate(1, a.rows())).cwiseMin(b.replicate(b.cols(), 1));
        a = bboxes1.col(3);
        b = bboxes2.col(3);
        Eigen::MatrixXd yy2 = (a.replicate(1, a.rows())).cwiseMin(b.replicate(b.cols(), 1));
        Eigen::MatrixXd w = (xx2 - xx1).cwiseMax(0);
        Eigen::MatrixXd h = (yy2 - yy1).cwiseMax(0);
        Eigen::MatrixXd wh = w.array() * h.array();
        //    Eigen::MatrixXd part1= (bboxes1.col(2) - bboxes1.col(0)).array()*(bboxes1.col(3) - bboxes1.col(1)).array();
        a = (bboxes1.col(2) - bboxes1.col(0)).array() * (bboxes1.col(3) - bboxes1.col(1)).array();
        //    Eigen::Matrix<double,1,Eigen::Dynamic> part2= (bboxes2.col(2) - bboxes2.col(0)).array()*(bboxes2.col(3) - bboxes2.col(1)).array();
        b = (bboxes2.col(2) - bboxes2.col(0)).array() * (bboxes2.col(3) - bboxes2.col(1)).array();
        // 做加法，但是还需要广播一下
        Eigen::MatrixXd part1_ = a.replicate(1, a.rows());
        Eigen::MatrixXd part2_ = b.replicate(b.cols(), 1);
        Eigen::MatrixXd Sum = part1_ + part2_ - wh;
        return wh.cwiseQuotient(Sum);
    }

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ocsort::associate(Eigen::MatrixXd detections, Eigen::MatrixXd trackers, float iou_threshold, Eigen::MatrixXd velocities, Eigen::MatrixXd previous_obs, float vdc_weight) {
        if (trackers.rows() == 0) return std::make_tuple(Eigen::VectorXd::Zero(0, 2), Eigen::VectorXd::LinSpaced(detections.rows(), 0, detections.rows() - 1), Eigen::VectorXd::Zero(0, 5));

        Eigen::MatrixXd Y, X;
        auto result = speed_direction_batch(detections, previous_obs);
        Y = std::get<0>(result);
        X = std::get<1>(result);
        Eigen::MatrixXd inertia_Y = velocities.col(0);
        Eigen::MatrixXd inertia_X = velocities.col(1);
        // todo 这里maybe会有问题
        inertia_Y = inertia_Y.transpose().replicate(1, inertia_Y.cols());
        inertia_X = inertia_X.transpose().replicate(1, inertia_X.cols());
        // 然而这里和numpy不一样，eigen中需要显式声明用 逐元素点乘
        Eigen::MatrixXd diff_angle_cos = inertia_X.array() * X.array() + inertia_Y.array() * Y.array();
        // 计算了两个向量的余弦后，应该将他们的值限制在 -1 ~ 1 之间，保证计算的正确性
        diff_angle_cos = (diff_angle_cos.array().min(1).max(-1)).matrix();
        // 计算反余弦值
        Eigen::MatrixXd diff_angle = Eigen::acos(diff_angle_cos.array());
        // fixme？这个掩码是拿来干嘛的？
        Eigen::Array<bool, 1, Eigen::Dynamic> valid_mask = Eigen::Array<bool, Eigen::Dynamic, 1>::Ones(previous_obs.rows());
        valid_mask.colwise() *= (previous_obs.col(4).array() >= 0);// 然后numpy中，这里应该是个行向量的，但是C++中却变成了列向量
        Eigen::MatrixXd iou_matrix = iou_batch(detections, trackers);
        Eigen::MatrixXd scores = detections.col(detections.cols() - 1);// 取最后一列,fixme:置信度吧好像是


        return std::make_tuple(X, Y, valid_mask);
    }
}// namespace ocsort