#include "../include/association.h"
#include <iomanip>
#include <iostream>

namespace ocsort {
    std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> speed_direction_batch(const Eigen::MatrixXf &dets,
                                                                       const Eigen::MatrixXf &tracks) {
        Eigen::VectorXf CX1 = (dets.col(0) + dets.col(2)) / 2.0;
        Eigen::VectorXf CY1 = (dets.col(1) + dets.col(3)) / 2.f;
        Eigen::MatrixXf CX2 = (tracks.col(0) + tracks.col(2)) / 2.f;
        Eigen::MatrixXf CY2 = (tracks.col(1) + tracks.col(3)) / 2.f;
        Eigen::MatrixXf dx = CX1.transpose().replicate(tracks.rows(), 1) - CX2.replicate(1, dets.rows());
        Eigen::MatrixXf dy = CY1.transpose().replicate(tracks.rows(), 1) - CY2.replicate(1, dets.rows());
        Eigen::MatrixXf norm = (dx.array().square() + dy.array().square()).sqrt() + 1e-6f;
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
    Eigen::MatrixXf iou_batch(const Eigen::MatrixXf &bboxes1, const Eigen::MatrixXf &bboxes2) {
        Eigen::Matrix<float, Eigen::Dynamic, 1> a = bboxes1.col(0);// bboxes1[..., 0] (n1,1)
        Eigen::Matrix<float, 1, Eigen::Dynamic> b = bboxes2.col(0);// bboxes2[..., 0] (1,n2)
        Eigen::MatrixXf xx1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
        a = bboxes1.col(1);// bboxes1[..., 1]
        b = bboxes2.col(1);// bboxes2[..., 1]
        Eigen::MatrixXf yy1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
        a = bboxes1.col(2);// bboxes1[..., 2]
        b = bboxes2.col(2);// bboxes1[..., 2]
        Eigen::MatrixXf xx2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
        a = bboxes1.col(3);// bboxes1[..., 3]
        b = bboxes2.col(3);// bboxes1[..., 3]
        Eigen::MatrixXf yy2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
        Eigen::MatrixXf w = (xx2 - xx1).cwiseMax(0);
        Eigen::MatrixXf h = (yy2 - yy1).cwiseMax(0);
        Eigen::MatrixXf wh = w.array() * h.array();
        a = (bboxes1.col(2) - bboxes1.col(0)).array() * (bboxes1.col(3) - bboxes1.col(1)).array();
        b = (bboxes2.col(2) - bboxes2.col(0)).array() * (bboxes2.col(3) - bboxes2.col(1)).array();
        // Addition with broadcasting
        Eigen::MatrixXf part1_ = a.replicate(1, b.cols());
        Eigen::MatrixXf part2_ = b.replicate(a.rows(), 1);
        Eigen::MatrixXf Sum = part1_ + part2_ - wh;// Shape: (n1,n2)
        return wh.cwiseQuotient(Sum);
    }
    /**
     *
     * @param bboxes1 形状： (n1,6)
     * @param bboxes2 形状： (n2,5)
     * @return 形状：(n1,n2)
     */
    Eigen::MatrixXf giou_batch(const Eigen::MatrixXf &bboxes1, const Eigen::MatrixXf &bboxes2) {
        Eigen::Matrix<float, Eigen::Dynamic, 1> a = bboxes1.col(0);// bboxes1[..., 0] (n1,1)
        Eigen::Matrix<float, 1, Eigen::Dynamic> b = bboxes2.col(0);// bboxes2[..., 0] (1,n2)
        Eigen::MatrixXf xx1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
        a = bboxes1.col(1);// bboxes1[..., 1]
        b = bboxes2.col(1);// bboxes2[..., 1]
        Eigen::MatrixXf yy1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
        a = bboxes1.col(2);// bboxes1[..., 2]
        b = bboxes2.col(2);// bboxes1[..., 2]
        Eigen::MatrixXf xx2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
        a = bboxes1.col(3);// bboxes1[..., 3]
        b = bboxes2.col(3);// bboxes1[..., 3]
        Eigen::MatrixXf yy2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
        Eigen::MatrixXf w = (xx2 - xx1).cwiseMax(0);
        Eigen::MatrixXf h = (yy2 - yy1).cwiseMax(0);
        Eigen::MatrixXf wh = w.array() * h.array();
        a = (bboxes1.col(2) - bboxes1.col(0)).array() * (bboxes1.col(3) - bboxes1.col(1)).array();
        b = (bboxes2.col(2) - bboxes2.col(0)).array() * (bboxes2.col(3) - bboxes2.col(1)).array();
        // Addition with broadcasting
        Eigen::MatrixXf part1_ = a.replicate(1, b.cols());
        Eigen::MatrixXf part2_ = b.replicate(a.rows(), 1);
        Eigen::MatrixXf Sum = part1_ + part2_ - wh;// Shape: (n1,n2)
        Eigen::MatrixXf iou = wh.cwiseQuotient(Sum);

        a = bboxes1.col(0);
        b = bboxes2.col(0);
        Eigen::MatrixXf xxc1 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
        a = bboxes1.col(1);// bboxes1[..., 1]
        b = bboxes2.col(1);// bboxes2[..., 1]
        Eigen::MatrixXf yyc1 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
        a = bboxes1.col(2);// bboxes1[..., 2]
        b = bboxes2.col(2);// bboxes1[..., 2]
        Eigen::MatrixXf xxc2 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
        a = bboxes1.col(3);// bboxes1[..., 3]
        b = bboxes2.col(3);// bboxes1[..., 3]
        Eigen::MatrixXf yyc2 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));

        Eigen::MatrixXf wc = xxc2 - xxc1;
        Eigen::MatrixXf hc = yyc2 - yyc1;
        // Simulate assert((wc > 0).all() and (hc > 0).all())
        if ((wc.array() > 0).all() && (hc.array() > 0).all())
            return iou;
        else {
            Eigen::MatrixXf area_enclose = wc.array() * hc.array();
            Eigen::MatrixXf giou = iou.array() - (area_enclose.array() - wh.array()) / area_enclose.array();
            giou = (giou.array() + 1) / 2.0;// Scale associateiou from (-1,1) to (0,1)
            return giou;
        }
    }

    std::tuple<std::vector<Eigen::Matrix<int, 1, 2>>, std::vector<int>, std::vector<int>> associate(Eigen::MatrixXf detections, Eigen::MatrixXf trackers, float iou_threshold, Eigen::MatrixXf velocities, Eigen::MatrixXf previous_obs_, float vdc_weight) {
        if (trackers.rows() == 0) {
            // If no trackers, return empty matches, but unmatched_dets is not empty.
            std::vector<int> unmatched_dets;
            for (int i = 0; i < detections.rows(); i++) {
                unmatched_dets.push_back(i);
            }
            return std::make_tuple(std::vector<Eigen::Matrix<int, 1, 2>>(), unmatched_dets, std::vector<int>());
        }
        Eigen::MatrixXf Y, X;
        auto result = speed_direction_batch(detections, previous_obs_);
        Y = std::get<0>(result);
#include "../include/association.h"
#include <iomanip>
#include <iostream>

        namespace ocsort {
            std::tuple<Eigen::MatrixXf, Eigen::MatrixXf> speed_direction_batch(const Eigen::MatrixXf &dets,
                                                                               const Eigen::MatrixXf &tracks) {
                Eigen::VectorXf CX1 = (dets.col(0) + dets.col(2)) / 2.0;
                Eigen::VectorXf CY1 = (dets.col(1) + dets.col(3)) / 2.f;
                Eigen::MatrixXf CX2 = (tracks.col(0) + tracks.col(2)) / 2.f;
                Eigen::MatrixXf CY2 = (tracks.col(1) + tracks.col(3)) / 2.f;
                Eigen::MatrixXf dx = CX1.transpose().replicate(tracks.rows(), 1) - CX2.replicate(1, dets.rows());
                Eigen::MatrixXf dy = CY1.transpose().replicate(tracks.rows(), 1) - CY2.replicate(1, dets.rows());
                Eigen::MatrixXf norm = (dx.array().square() + dy.array().square()).sqrt() + 1e-6f;
                dx = dx.array() / norm.array();
                dy = dy.array() / norm.array();
                return std::make_tuple(dy, dx);
            }
            /**
     *
     * @param bboxes1 Shape: (n1,6)
     * @param bboxes2 Shape: (n2,5)
     * @return Shape: (n1,n2)
     */
            Eigen::MatrixXf iou_batch(const Eigen::MatrixXf &bboxes1, const Eigen::MatrixXf &bboxes2) {
                Eigen::Matrix<float, Eigen::Dynamic, 1> a = bboxes1.col(0);// bboxes1[..., 0] (n1,1)
                Eigen::Matrix<float, 1, Eigen::Dynamic> b = bboxes2.col(0);// bboxes2[..., 0] (1,n2)
                Eigen::MatrixXf xx1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
                a = bboxes1.col(1);// bboxes1[..., 1]
                b = bboxes2.col(1);// bboxes2[..., 1]
                Eigen::MatrixXf yy1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
                a = bboxes1.col(2);// bboxes1[..., 2]
                b = bboxes2.col(2);// bboxes1[..., 2]
                Eigen::MatrixXf xx2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
                a = bboxes1.col(3);// bboxes1[..., 3]
                b = bboxes2.col(3);// bboxes1[..., 3]
                Eigen::MatrixXf yy2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
                Eigen::MatrixXf w = (xx2 - xx1).cwiseMax(0);
                Eigen::MatrixXf h = (yy2 - yy1).cwiseMax(0);
                Eigen::MatrixXf wh = w.array() * h.array();
                a = (bboxes1.col(2) - bboxes1.col(0)).array() * (bboxes1.col(3) - bboxes1.col(1)).array();
                b = (bboxes2.col(2) - bboxes2.col(0)).array() * (bboxes2.col(3) - bboxes2.col(1)).array();
                // Addition with broadcasting
                Eigen::MatrixXf part1_ = a.replicate(1, b.cols());
                Eigen::MatrixXf part2_ = b.replicate(a.rows(), 1);
                Eigen::MatrixXf Sum = part1_ + part2_ - wh;// Shape: (n1,n2)
                return wh.cwiseQuotient(Sum);
            }
            /**
     *
     * @param bboxes1 Shape: (n1,6)
     * @param bboxes2 Shape: (n2,5)
     * @return Shape: (n1,n2)
     */
            Eigen::MatrixXf giou_batch(const Eigen::MatrixXf &bboxes1, const Eigen::MatrixXf &bboxes2) {
                Eigen::Matrix<float, Eigen::Dynamic, 1> a = bboxes1.col(0);// bboxes1[..., 0] (n1,1)
                Eigen::Matrix<float, 1, Eigen::Dynamic> b = bboxes2.col(0);// bboxes2[..., 0] (1,n2)
                Eigen::MatrixXf xx1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
                a = bboxes1.col(1);// bboxes1[..., 1]
                b = bboxes2.col(1);// bboxes2[..., 1]
                Eigen::MatrixXf yy1 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
                a = bboxes1.col(2);// bboxes1[..., 2]
                b = bboxes2.col(2);// bboxes1[..., 2]
                Eigen::MatrixXf xx2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
                a = bboxes1.col(3);// bboxes1[..., 3]
                b = bboxes2.col(3);// bboxes1[..., 3]
                Eigen::MatrixXf yy2 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
                Eigen::MatrixXf w = (xx2 - xx1).cwiseMax(0);
                Eigen::MatrixXf h = (yy2 - yy1).cwiseMax(0);
                Eigen::MatrixXf wh = w.array() * h.array();
                a = (bboxes1.col(2) - bboxes1.col(0)).array() * (bboxes1.col(3) - bboxes1.col(1)).array();
                b = (bboxes2.col(2) - bboxes2.col(0)).array() * (bboxes2.col(3) - bboxes2.col(1)).array();
                // Addition with broadcasting
                Eigen::MatrixXf part1_ = a.replicate(1, b.cols());
                Eigen::MatrixXf part2_ = b.replicate(a.rows(), 1);
                Eigen::MatrixXf Sum = part1_ + part2_ - wh;// Shape: (n1,n2)
                Eigen::MatrixXf iou = wh.cwiseQuotient(Sum);

                a = bboxes1.col(0);
                b = bboxes2.col(0);
                Eigen::MatrixXf xxc1 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
                a = bboxes1.col(1);// bboxes1[..., 1]
                b = bboxes2.col(1);// bboxes2[..., 1]
                Eigen::MatrixXf yyc1 = (a.replicate(1, b.cols())).cwiseMin(b.replicate(a.rows(), 1));
                a = bboxes1.col(2);// bboxes1[..., 2]
                b = bboxes2.col(2);// bboxes1[..., 2]
                Eigen::MatrixXf xxc2 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));
                a = bboxes1.col(3);// bboxes1[..., 3]
                b = bboxes2.col(3);// bboxes1[..., 3]
                Eigen::MatrixXf yyc2 = (a.replicate(1, b.cols())).cwiseMax(b.replicate(a.rows(), 1));

                Eigen::MatrixXf wc = xxc2 - xxc1;
                Eigen::MatrixXf hc = yyc2 - yyc1;
                // Simulate assert((wc > 0).all() and (hc > 0).all())
                if ((wc.array() > 0).all() && (hc.array() > 0).all())
                    return iou;
                else {
                    Eigen::MatrixXf area_enclose = wc.array() * hc.array();
                    Eigen::MatrixXf giou = iou.array() - (area_enclose.array() - wh.array()) / area_enclose.array();
                    giou = (giou.array() + 1) / 2.0;// Scale associateiou from (-1,1) to (0,1)
                    return giou;
                }
            }

            std::tuple<std::vector<Eigen::Matrix<int, 1, 2>>, std::vector<int>, std::vector<int>> associate(Eigen::MatrixXf detections, Eigen::MatrixXf trackers, float iou_threshold, Eigen::MatrixXf velocities, Eigen::MatrixXf previous_obs_, float vdc_weight) {
                if (trackers.rows() == 0) {
                    // If no trackers, return empty matches, but unmatched_dets is not empty.
                    std::vector<int> unmatched_dets;
                    for (int i = 0; i < detections.rows(); i++) {
                        unmatched_dets.push_back(i);
                    }
                    return std::make_tuple(std::vector<Eigen::Matrix<int, 1, 2>>(), unmatched_dets, std::vector<int>());
                }
                Eigen::MatrixXf Y, X;
                auto result = speed_direction_batch(detections, previous_obs_);
                Y = std::get<0>(result);
                X = std::get<1>(result);
                Eigen::MatrixXf inertia_Y = velocities.col(0);// Shape is (2,1), transpose needed for broadcasting
                Eigen::MatrixXf inertia_X = velocities.col(1);
                Eigen::MatrixXf inertia_Y_ = inertia_Y.replicate(1, Y.cols());
                Eigen::MatrixXf inertia_X_ = inertia_X.replicate(1, X.cols());
                Eigen::MatrixXf diff_angle_cos = inertia_X_.array() * X.array() + inertia_Y_.array() * Y.array();
                // After calculating cosine of two vectors, limit values to -1 ~ 1 for correctness
                diff_angle_cos = diff_angle_cos.array().cwiseMin(1.0f).cwiseMax(-1.0f).matrix();
                // Calculate arccos
                Eigen::MatrixXf diff_angle = Eigen::acos(diff_angle_cos.array());
                // fixme: Almost forgot this step: diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi
                diff_angle = (pi / 2.0 - diff_angle.array().abs()).array() / (pi);
                Eigen::Array<bool, 1, Eigen::Dynamic> valid_mask = Eigen::Array<bool, Eigen::Dynamic, 1>::Ones(previous_obs_.rows());
                valid_mask = valid_mask.array() * ((previous_obs_.col(4).array() >= 0).transpose()).array();// In numpy, this should be a row vector
                Eigen::MatrixXf iou_matrix = iou_batch(detections, trackers);
                Eigen::MatrixXf scores = detections.col(detections.cols() - 2).replicate(1, trackers.rows());// Take the second to last column, fixme: seems to be confidence
                Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask_ = (valid_mask.transpose()).replicate(1, X.cols());
                Eigen::MatrixXf angle_diff_cost;
                auto valid_float = valid_mask_.cast<float>();
                auto intermediate_result = (valid_float.array() * diff_angle.array() * vdc_weight).transpose();
                angle_diff_cost.noalias() = (intermediate_result.array() * scores.array()).matrix();
                Eigen::Matrix<float, Eigen::Dynamic, 2> matched_indices(0, 2);// Unlike python, this local variable should be defined here
                if (std::min(iou_matrix.cols(), iou_matrix.rows()) > 0) {
                    Eigen::MatrixXf a = (iou_matrix.array() > iou_threshold).cast<float>();
                    float sum1 = (a.rowwise().sum()).maxCoeff();
                    float sum0 = (a.colwise().sum()).maxCoeff();
                    // float and int cannot be compared directly with ==, simulate a.sum(1).max() == 1 and a.sum(0).max()
                    if ((fabs(sum1 - 1) < 1e-12) && (fabs(sum0 - 1) < 1e-12)) {
                        // The following operation simulates: np.stack(np.where(a), axis=1)
                        for (int i = 0; i < a.rows(); i++) {
                            for (int j = 0; j < a.cols(); j++) {
                                if (a(i, j) > 0) {
                                    Eigen::RowVectorXf row(2);
                                    row << i, j;
                                    matched_indices.conservativeResize(matched_indices.rows() + 1, Eigen::NoChange);
                                    matched_indices.row(matched_indices.rows() - 1) = row;
                                }
                            }
                        }
                    } else {
                        // note: Linear assignment
                        Eigen::MatrixXf cost_matrix = iou_matrix.array() + angle_diff_cost.array();
                        // Convert to 2D vector
                        std::vector<std::vector<float>> cost_iou_matrix(cost_matrix.rows(), std::vector<float>(cost_matrix.cols()));
                        for (int i = 0; i < cost_matrix.rows(); i++) {
                            for (int j = 0; j < cost_matrix.cols(); j++) {
                                cost_iou_matrix[i][j] = -cost_matrix(i, j);// note: Negate here
                            }
                        }
                        // Perform linear assignment
                        std::vector<int> rowsol, colsol;
                        float MIN_cost = execLapjv(cost_iou_matrix, rowsol, colsol, true, 0.01, true);
                        for (int i = 0; i < rowsol.size(); i++) {
                            if (rowsol.at(i) >= 0) {
                                Eigen::RowVectorXf row(2);
                                row << colsol.at(rowsol.at(i)), rowsol.at(i);
                                matched_indices.conservativeResize(matched_indices.rows() + 1, Eigen::NoChange);
                                matched_indices.row(matched_indices.rows() - 1) = row;
                            }
                        }
                    }
                } else {
                    matched_indices = Eigen::MatrixXf(0, 2);// Otherwise return empty (0,2) matrix
                }
                // fixme: This is Z^{remain}_t, detections not matched to tracks
                std::vector<int> unmatched_detections;// Indices of unmatched observations
                for (int i = 0; i < detections.rows(); i++) {
                    if ((matched_indices.col(0).array() == i).sum() == 0) {
                        unmatched_detections.push_back(i);
                    }
                }
                // fixme: This is T^{remain}_t, tracks not matched to detections
                std::vector<int> unmatched_trackers;// Also recording indices
                for (int i = 0; i < trackers.rows(); i++) {
                    if ((matched_indices.col(1).array() == i).sum() == 0) {
                        unmatched_trackers.push_back(i);
                    }
                }
                ///////////////
                /// filter out matched with low IOU
                //////////////
                std::vector<Eigen::Matrix<int, 1, 2>> matches;
                // Iterate matched_indices by row
                Eigen::Matrix<int, 1, 2> tmp;// Actually shape is fixed (1,2)
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
                    // Should return empty vector, so clear after initialization?
                    matches.clear();
                }
                /**
           * Returns matches => vector<Eigen::Matrix<int,1,2>> array
           * unmatched_detections => vector<int> Indices of detections not matched to tracks
           * unmatched_trackers => vector<int> Indices of tracks not matched to detections
       */
                return std::make_tuple(matches, unmatched_detections, unmatched_trackers);
            }
        }// namespace ocsort