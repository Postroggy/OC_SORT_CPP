#include "../include/association.h"
#include <iomanip>
#include <iostream>

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

    std::tuple<std::vector<Eigen::Matrix<int, 1, Eigen::Dynamic>>, std::vector<int>, std::vector<int>> ocsort::associate(Eigen::MatrixXd detections, Eigen::MatrixXd trackers, float iou_threshold, Eigen::MatrixXd velocities, Eigen::MatrixXd previous_obs, float vdc_weight) {
        if (trackers.rows() == 0) {
            // 如果 tracker 没有的话，直接返回空的，但是 unmatched_dets不为空。
            std::vector<int> unmatched_dets;
            for(int i=0;i<detections.rows();i++){
                unmatched_dets.push_back(i);
            }
            return std::make_tuple(std::vector<Eigen::Matrix<int, 1, Eigen::Dynamic>>(), unmatched_dets, std::vector<int>());
        }
        Eigen::MatrixXd Y, X;
        auto result = speed_direction_batch(detections, previous_obs);
        Y = std::get<0>(result);
        X = std::get<1>(result);
        Eigen::MatrixXd inertia_Y = velocities.col(0);// shape是 (2,1)的，后续做广播需要逆置一下
        Eigen::MatrixXd inertia_X = velocities.col(1);
        // todo 这里maybe会有问题, 我这里又重新声明了一个变量保存值，因为不能直接 a = a.transpose()
        Eigen::MatrixXd inertia_Y_ = inertia_Y.replicate(1, Y.cols());
        Eigen::MatrixXd inertia_X_ = inertia_X.replicate(1, X.cols());
        // fixme:这条代码有问题，明天早上27号，起来搞 :然而这里和numpy不一样，eigen中需要显式声明用 逐元素点乘
        Eigen::MatrixXd diff_angle_cos = inertia_X_.array() * X.array() + inertia_Y_.array() * Y.array();
        // 计算了两个向量的余弦后，应该将他们的值限制在 -1 ~ 1 之间，保证计算的正确性
        diff_angle_cos = (diff_angle_cos.array().min(1).max(-1)).matrix();
        // 计算反余弦值
        Eigen::MatrixXd diff_angle = Eigen::acos(diff_angle_cos.array());
        // fixme: 差点忘记这一步了： diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi
        diff_angle = (pi / 2.0 - diff_angle.array().abs()).array() / (pi);
        // fixme？这个掩码是拿来干嘛的？
        Eigen::Array<bool, 1, Eigen::Dynamic> valid_mask = Eigen::Array<bool, Eigen::Dynamic, 1>::Ones(previous_obs.rows());
        previous_obs(0, 4) = 9;
        valid_mask = valid_mask.array() * ((previous_obs.col(4).array() >= 0).transpose()).array();// 然后numpy中，这里应该是个行向量的
        Eigen::MatrixXd iou_matrix = iou_batch(detections, trackers);
        Eigen::MatrixXd scores = detections.col(detections.cols() - 2).replicate(1, trackers.rows());// 取倒数第二列,fixme:置信度吧好像是
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask_ = (valid_mask.transpose()).replicate(1, X.cols());
        Eigen::MatrixXd angle_diff_cost = ((((valid_mask_.cast<double>()).array() * diff_angle.array()).array() * vdc_weight)
                                                   .transpose())
                                                  .array() *
                                          scores.array();
        // fixme 个人感觉这个判断没什么用
        Eigen::MatrixXd matched_indices(0, 2);// 跟python不一样，这个局部变量应该定义在这里
        if (std::min(iou_matrix.cols(), iou_matrix.rows()) > 0) {
            Eigen::MatrixXd a = (iou_matrix.array() > iou_threshold).cast<double>();
            double sum1 = (a.rowwise().sum()).maxCoeff();
            double sum0 = (a.colwise().sum()).maxCoeff();
            // double 和 int 不能直接 == 比较 ,模拟a.sum(1).max() == 1 and a.sum(0).max()
            if (fabs(sum1 - 1) < 1e-12 && fabs(sum0 - 1) < 1e-12) {
                // todo: 这里可能会出bug，没看懂他这儿 np.stack 能干嘛
                // 下面的操作是模拟：np.stack(np.where(a), axis=1)
                for (int i = 0; i < a.rows(); i++) {
                    for (int j = 0; j < a.cols(); j++) {
                        if (a(i, j) > 0) {
                            Eigen::RowVectorXd row(2);
                            row << i, j;
                            matched_indices.conservativeResize(matched_indices.rows() + 1, Eigen::NoChange);
                            matched_indices.row(matched_indices.rows() - 1) = row;
                        }
                    }
                }
            } else {
                // todo: 等下再写了
                //                Eigen::Matrix matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost));
            }
        } else {
            matched_indices = Eigen::MatrixXd(0, 2);// 否则就返回空的 (0,2) 的矩阵
        }
        // fixme: 这是 Z^{remain}_t ，没有匹配上轨迹的 检测
        std::vector<int> unmatched_detections;// 这是未匹配上的 观测值的 index
        for (int i = 0; i < detections.rows(); i++) {
            // 模拟 if(d not in matched_indices[:,0]), 注意是 not in
            if ((matched_indices.col(0).array() == i).sum() == 0) {
                unmatched_detections.push_back(i);
            }
        }
        // fixme: 这是 T^{remain}_t ,没有匹配上检测的 轨迹
        std::vector<int> unmatched_trackers;// 也是记录 index
        for (int i = 0; i < trackers.rows(); i++) {
            // 模拟 if(d not in matched_indices[:,1]), 注意是 not in
            if ((matched_indices.col(1).array() == i).sum() == 0) {
                unmatched_detections.push_back(i);
            }
        }
        ///////////////
        /// filter out matched with low IOU
        //////////////
        std::vector<Eigen::Matrix<int, 1, Eigen::Dynamic>> matches;
        // 按行遍历matched_indices
        Eigen::Matrix<int, 1, Eigen::Dynamic> tmp;// 实际上形状是固定的 (1,2)
        for (int i = 0; i < matched_indices.rows(); i++) {
            tmp = (matched_indices.row(i)).cast<int>();
            if (iou_matrix(tmp(0), tmp(1)) < iou_threshold) {
                unmatched_detections.push_back(tmp(0));
                unmatched_trackers.push_back(tmp(1));
            } else {
                matches.push_back(tmp);
            }
        }
        if (matches.size() == 0) {
            // todo ? 这里会报错嘛？
            // 应该是要返回空的vector的，所有初始化之后，clear一下？
            matches.clear();
        }
        /**
           * 返回的是 matches => vector<Eigen::Matrix<int,1,2>> 的数组
           * unmatched_detections => vector<int> 代表没匹配上轨迹的观测值的index
           * unmatched_trackers => vector<int> 代表没匹配上观测值的轨迹的index
       */
        return std::make_tuple(matches, unmatched_detections, unmatched_trackers);
    }
}// namespace ocsort