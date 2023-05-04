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
        //        std::cout << "==============START speed_direction_batch DEBUG =================" << std::endl;
        //        std::cout << "CX1\n"
        //                  << CX1 << "\nCY1\n"
        //                  << CY1 << "\nCX2\n"
        //                  << CX2 << "\nCY2\n"
        //                  << CY2 << std::endl;
        //        std::cout << "==============ENDs speed_direction_batch DEBUG =================" << std::endl;
        // todo: :WARNING: 这里报错了，
        Eigen::MatrixXd dx = CX1.transpose().replicate(tracks.rows(), 1) - CX2.replicate(1, dets.rows());
        Eigen::MatrixXd dy = CY1.transpose().replicate(tracks.rows(), 1) - CY2.replicate(1, dets.rows());
        Eigen::MatrixXd norm = (dx.array().square() + dy.array().square()).sqrt() + 1e-6f;
        dx = dx.array() / norm.array();
        dy = dy.array() / norm.array();
        return std::make_tuple(dy, dx);
    }
    /**
     *
     * @param bboxes1 形状： (n1,6)
     * @param bboxes2 形状： (n2,5)
     * @return 形状：(n1,n2)
     */
    Eigen::MatrixXd iou_batch(const Eigen::MatrixXd &bboxes1, const Eigen::MatrixXd &bboxes2) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> a = bboxes1.col(0);// bboxes1[..., 0] (n1,1)
        Eigen::Matrix<double, 1, Eigen::Dynamic> b = bboxes2.col(0);// bboxes2[..., 0] (1,n2)
        Eigen::MatrixXd xx1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
        a = bboxes1.col(1);// bboxes1[..., 1]
        b = bboxes2.col(1);// bboxes2[..., 1]
        Eigen::MatrixXd yy1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
        a = bboxes1.col(2);// bboxes1[..., 2]
        b = bboxes2.col(2);// bboxes1[..., 2]
        Eigen::MatrixXd xx2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
        a = bboxes1.col(3);// bboxes1[..., 3]
        b = bboxes2.col(3);// bboxes1[..., 3]
        Eigen::MatrixXd yy2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
        Eigen::MatrixXd w = (xx2 - xx1).cwiseMax(0);
        Eigen::MatrixXd h = (yy2 - yy1).cwiseMax(0);
        Eigen::MatrixXd wh = w.array() * h.array();
        //    Eigen::MatrixXd part1= (bboxes1.col(2) - bboxes1.col(0)).array()*(bboxes1.col(3) - bboxes1.col(1)).array();
        a = (bboxes1.col(2) - bboxes1.col(0)).array() * (bboxes1.col(3) - bboxes1.col(1)).array();
        //    Eigen::Matrix<double,1,Eigen::Dynamic> part2= (bboxes2.col(2) - bboxes2.col(0)).array()*(bboxes2.col(3) - bboxes2.col(1)).array();
        b = (bboxes2.col(2) - bboxes2.col(0)).array() * (bboxes2.col(3) - bboxes2.col(1)).array();
        //        std::cout << "Current Line:" << __LINE__ << std::endl;
        //        std::cout << "a :\n"
        //                  << a << "\nb:\n"
        //                  << b << std::endl;
        // 做加法，但是还需要广播一下
        Eigen::MatrixXd part1_ = a.replicate(1, b.cols());
        Eigen::MatrixXd part2_ = b.replicate(a.rows(), 1);
        Eigen::MatrixXd Sum = part1_ + part2_ - wh;// 形状：(n1,n2)
        return wh.cwiseQuotient(Sum);
    }
    /**
     *
     * @param bboxes1 形状： (n1,6)
     * @param bboxes2 形状： (n2,5)
     * @return 形状：(n1,n2)
     */
    Eigen::MatrixXd giou_batch(const Eigen::MatrixXd &bboxes1, const Eigen::MatrixXd &bboxes2) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> a = bboxes1.col(0);// bboxes1[..., 0] (n1,1)
        Eigen::Matrix<double, 1, Eigen::Dynamic> b = bboxes2.col(0);// bboxes2[..., 0] (1,n2)
        Eigen::MatrixXd xx1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
        a = bboxes1.col(1);// bboxes1[..., 1]
        b = bboxes2.col(1);// bboxes2[..., 1]
        Eigen::MatrixXd yy1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
        a = bboxes1.col(2);// bboxes1[..., 2]
        b = bboxes2.col(2);// bboxes1[..., 2]
        Eigen::MatrixXd xx2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
        a = bboxes1.col(3);// bboxes1[..., 3]
        b = bboxes2.col(3);// bboxes1[..., 3]
        Eigen::MatrixXd yy2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
        Eigen::MatrixXd w = (xx2 - xx1).cwiseMax(0);
        Eigen::MatrixXd h = (yy2 - yy1).cwiseMax(0);
        Eigen::MatrixXd wh = w.array() * h.array();
        a = (bboxes1.col(2) - bboxes1.col(0)).array() * (bboxes1.col(3) - bboxes1.col(1)).array();
        b = (bboxes2.col(2) - bboxes2.col(0)).array() * (bboxes2.col(3) - bboxes2.col(1)).array();
        // 做加法，但是还需要广播一下
        Eigen::MatrixXd part1_ = a.replicate(1, b.cols());
        Eigen::MatrixXd part2_ = b.replicate(a.rows(), 1);
        Eigen::MatrixXd Sum = part1_ + part2_ - wh;// 形状：(n1,n2)
        Eigen::MatrixXd iou = wh.cwiseQuotient(Sum);

        a = bboxes1.col(0);
        b = bboxes2.col(0);
        Eigen::MatrixXd xxc1 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
        a = bboxes1.col(1);// bboxes1[..., 1]
        b = bboxes2.col(1);// bboxes2[..., 1]
        Eigen::MatrixXd yyc1 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
        a = bboxes1.col(2);// bboxes1[..., 2]
        b = bboxes2.col(2);// bboxes1[..., 2]
        Eigen::MatrixXd xxc2 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
        a = bboxes1.col(3);// bboxes1[..., 3]
        b = bboxes2.col(3);// bboxes1[..., 3]
        Eigen::MatrixXd yyc2 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));

        Eigen::MatrixXd wc = xxc2 - xxc1;
        Eigen::MatrixXd hc = yyc2 - yyc1;
        // 模拟 assert((wc > 0).all() and (hc > 0).all())
        if ((wc.array() > 0).all() && (hc.array() > 0).all())
            return iou;
        else {
            Eigen::MatrixXd area_enclose = wc.array() * hc.array();
            Eigen::MatrixXd giou = iou.array() - (area_enclose.array() - wh.array()) / area_enclose.array();
            giou = (giou.array() + 1) / 2.0;// 从 (-1,1) 缩放到 (0,1)
            return giou;
        }
    }

    std::tuple<std::vector<Eigen::Matrix<int, 1, 2>>, std::vector<int>, std::vector<int>> associate(Eigen::MatrixXd detections, Eigen::MatrixXd trackers, float iou_threshold, Eigen::MatrixXd velocities, Eigen::MatrixXd previous_obs_, float vdc_weight) {
        if (trackers.rows() == 0) {
            // 如果 tracker 没有的话，直接返回空的，但是 unmatched_dets 不为空。
            std::vector<int> unmatched_dets;
            for (int i = 0; i < detections.rows(); i++) {
                unmatched_dets.push_back(i);
            }
            return std::make_tuple(std::vector<Eigen::Matrix<int, 1, 2>>(), unmatched_dets, std::vector<int>());
        }
        Eigen::MatrixXd Y, X;
        auto result = speed_direction_batch(detections, previous_obs_);
        Y = std::get<0>(result);
        X = std::get<1>(result);
        //        std::cout << "X\n"
        //                  << X << std::endl;
        //        std::cout << "Y\n"
        //                  << Y << std::endl;
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
        Eigen::Array<bool, 1, Eigen::Dynamic> valid_mask = Eigen::Array<bool, Eigen::Dynamic, 1>::Ones(previous_obs_.rows());
        // note: 调试用
        // previous_obs_(0, 4) = 9;
        valid_mask = valid_mask.array() * ((previous_obs_.col(4).array() >= 0).transpose()).array();// 然后numpy中，这里应该是个行向量的
        Eigen::MatrixXd iou_matrix = iou_batch(detections, trackers);
        Eigen::MatrixXd scores = detections.col(detections.cols() - 2).replicate(1, trackers.rows());// 取倒数第二列,fixme:置信度吧好像是
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask_ = (valid_mask.transpose()).replicate(1, X.cols());
        Eigen::MatrixXd angle_diff_cost = ((((valid_mask_.cast<double>()).array() * diff_angle.array()).array() * vdc_weight)
                                                   .transpose())
                                                  .array() *
                                          scores.array();
        // TMD， 被这个局部变量坑惨了
        Eigen::Matrix<double, Eigen::Dynamic, 2> matched_indices(0, 2);// 跟python不一样，这个局部变量应该定义在这里
                                                                       //        std::vector<std::vector<int>> matched_indices;
        if (std::min(iou_matrix.cols(), iou_matrix.rows()) > 0) {
            Eigen::MatrixXd a = (iou_matrix.array() > iou_threshold).cast<double>();
            double sum1 = (a.rowwise().sum()).maxCoeff();
            double sum0 = (a.colwise().sum()).maxCoeff();
            // double 和 int 不能直接 == 比较 ,模拟a.sum(1).max() == 1 and a.sum(0).max()
            if ((fabs(sum1 - 1) < 1e-12) && (fabs(sum0 - 1) < 1e-12)) {
                // todo: 这里可能会出bug，没看懂他这儿 np.stack 能干嘛
                // 下面的操作是模拟：np.stack(np.where(a), axis=1)
                for (int i = 0; i < a.rows(); i++) {
                    for (int j = 0; j < a.cols(); j++) {
                        if (a(i, j) > 0) {
                            Eigen::RowVectorXd row(2);
                            row << i, j;
                            //                            matched_indices.push_back({i, j});
                            matched_indices.conservativeResize(matched_indices.rows() + 1, Eigen::NoChange);
                            matched_indices.row(matched_indices.rows() - 1) = row;
                        }
                    }
                }
            } else {
                // note: 线性分配
                Eigen::MatrixXd cost_matrix = iou_matrix.array() + angle_diff_cost.array();
                // 转化为二维vector
                std::vector<std::vector<double>> cost_iou_matrix(cost_matrix.rows(), std::vector<double>(cost_matrix.cols()));
                for (int i = 0; i < cost_matrix.rows(); i++) {
                    for (int j = 0; j < cost_matrix.cols(); j++) {
                        cost_iou_matrix[i][j] = -cost_matrix(i, j);// note： 这里取反
                    }
                }
                // 进行线性分配
                // todo : :WARNING: 主要问题这个 matched_indices 还是有问题
                std::vector<int> rowsol, colsol;
                double MIN_cost = execLapjv(cost_iou_matrix, rowsol, colsol, true, 0.01, true);
                // 生成 matched_indices , 实际上，使用 rowsol 就可以了
                // todo :WARNING: 找到问题所在了，外层初始化了一个 ，内层又初始化了一个名字相同的。
                //                matched_indices(rowsol.size(), std::vector<int>(2));
                // 为 matched_indices 赋值
                //                for (int i = 0; i < rowsol.size(); i++) {
                //                    Eigen::RowVectorXd row(2);
                //                    row << i, rowsol.at(i);
                //                    matched_indices.conservativeResize(matched_indices.rows() + 1, Eigen::NoChange);
                //                    matched_indices.row(matched_indices.rows() - 1) = row;
                //                }
                for (int i = 0; i < rowsol.size(); i++) {
                    if (rowsol.at(i) >= 0) {
                        //                        cout<<"==DEBUG== "<<"x[i]: "<<x.at(i)<<" y[x[i]]: "<<y.at(x.at(i))<<endl;
                        Eigen::RowVectorXd row(2);
                        row << colsol.at(rowsol.at(i)), rowsol.at(i);
                        matched_indices.conservativeResize(matched_indices.rows() + 1, Eigen::NoChange);
                        matched_indices.row(matched_indices.rows() - 1) = row;
                    }
                }
            }
        } else {
            matched_indices = Eigen::MatrixXd(0, 2);// 否则就返回空的 (0,2) 的矩阵
                                                    //            matched_indices.clear();
        }
        // fixme: 这是 Z^{remain}_t ，没有匹配上轨迹的 检测
        std::vector<int> unmatched_detections;// 这是未匹配上的 观测值的 index
        // todo :WARNING: 应该 不会出现 unmatched_detections 的呀。
        // note: 调试用
        //        std::cout << "BUD FOUND " << __LINE__ << std::endl;
        //        std::cout << "> detections.rows(): " << detections.rows() << std::endl;
        for (int i = 0; i < detections.rows(); i++) {
            // todo: 4/30号，为什么上一步已经match上了，这里还说事unmatched?
            //             模拟 if(d not in matched_indices[:,0]), 注意是 not in
            //            std::cout << "matched_indices.col(0):\n"
            //                      << matched_indices.col(0)
            //                      << " \ni :" << i
            //                      << "\nmatched_indices.col(0).array() == i\n"
            //                      << (matched_indices.col(0).array() == i)
            //                      << "(matched_indices.col(0).array() == i).sum()" << ((matched_indices.col(0).array() == i).sum()) << std::endl;
            if ((matched_indices.col(0).array() == i).sum() == 0) {
                //                std::cout << "FUCK" << std::endl;
                unmatched_detections.push_back(i);
            }
        }
        // fixme: 这是 T^{remain}_t ,没有匹配上检测的 轨迹
        std::vector<int> unmatched_trackers;// 也是记录 index
        for (int i = 0; i < trackers.rows(); i++) {
            // 模拟 if(d not in matched_indices[:,1]), 注意是 not in
            //            std::cout << "matched_indices.col(0):\n"
            //                      << matched_indices.col(1)
            //                      << " i :" << i << std::endl;
            if ((matched_indices.col(1).array() == i).sum() == 0) {
                unmatched_trackers.push_back(i);
            }
        }
        ///////////////
        /// filter out matched with low IOU
        //////////////
        std::vector<Eigen::Matrix<int, 1, 2>> matches;
        // 按行遍历 matched_indices
        Eigen::Matrix<int, 1, 2> tmp;// 实际上形状是固定的 (1,2)
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