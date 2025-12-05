#include "../include/OCsort.h"
#include "iomanip"
#include <utility>

namespace ocsort {
    /*Overload `<< for vector` to directly print the vector.*/
    template<typename Matrix>
    std::ostream &operator<<(std::ostream &os, const std::vector<Matrix> &v) {
        os << "{";
        for (auto it = v.begin(); it != v.end(); ++it) {
            os << "(" << *it << ")\n";
            if (it != v.end() - 1) os << ",";
        }
        os << "}\n";
        return os;
    }

    OCSort::OCSort(float det_thresh_, int max_age_, int min_hits_, float iou_threshold_, int delta_t_, std::string asso_func_, float inertia_, bool use_byte_) {
        /*Sets key parameters for SORT*/
        max_age = max_age_;
        min_hits = min_hits_;
        iou_threshold = iou_threshold_;
        trackers.clear();
        frame_count = 0;
        // ocsort newly added
        det_thresh = det_thresh_;
        delta_t = delta_t_;
        // Declare unordered_map, key is string, value is function pointer of function object without parameters and without return value
        std::unordered_map<std::string, std::function<Eigen::MatrixXf(const Eigen::MatrixXf &, const Eigen::MatrixXf &)>> ASSO_FUNCS{
                {"iou", iou_batch},
                {"giou", giou_batch}};
        // Determine the function to be used
        std::function<Eigen::MatrixXf(const Eigen::MatrixXf &, const Eigen::MatrixXf &)> asso_func = ASSO_FUNCS[asso_func_];
        inertia = inertia_;
        use_byte = use_byte_;
        KalmanBoxTracker::count = 0;
    }
    //! fixme: This is to control the printing precision, delete it when publishing
    std::ostream &precision(std::ostream &os) {
        os << std::fixed << std::setprecision(2);
        return os;
    }
    std::vector<Eigen::RowVectorXf> OCSort::update(Eigen::MatrixXf dets) {
        /*
         * Input matrix dets: shape (n,5) element format: [[x1,y1,x2,y2,confidence_score],...[...]]
         * Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        fixme：Original function prototype: def update(self, output_results, img_info, img_size)
            This part of the code is quite different, please refer to: https://www.diffchecker.com/fqTqcBSR/
         */
        frame_count += 1;
        /*Below are temporary variables*/
        Eigen::Matrix<float, Eigen::Dynamic, 4> xyxys = dets.leftCols(4);
        Eigen::Matrix<float, 1, Eigen::Dynamic> confs = dets.col(4);// This is a row vector of shape (1,n1)
        Eigen::Matrix<float, 1, Eigen::Dynamic> clss = dets.col(5); // The last column is the class of the target
        Eigen::MatrixXf output_results = dets;
        auto inds_low = confs.array() > 0.1;
        auto inds_high = confs.array() < det_thresh;
        // The confidence is between 0.1 and det_thresh, need to be matched twice inds_second => (1xN)
        auto inds_second = inds_low && inds_high;
        // Filter, simulate dets_second = output_results[inds_second];dets = output_results[remain_inds]
        Eigen::Matrix<float, Eigen::Dynamic, 6> dets_second;// Rows are not fixed, columns are fixed
        Eigen::Matrix<bool, 1, Eigen::Dynamic> remain_inds = (confs.array() > det_thresh);
        // Because the type of python can be changed at will while C++ cannot, note: Use dets_first to replace dets when passing to associate() function, remember!
        Eigen::Matrix<float, Eigen::Dynamic, 6> dets_first;
        for (int i = 0; i < output_results.rows(); i++) {
            if (true == inds_second(i)) {
                dets_second.conservativeResize(dets_second.rows() + 1, Eigen::NoChange);
                dets_second.row(dets_second.rows() - 1) = output_results.row(i);
            }
            if (true == remain_inds(i)) {
                dets_first.conservativeResize(dets_first.rows() + 1, Eigen::NoChange);
                dets_first.row(dets_first.rows() - 1) = output_results.row(i);
            }
        }
        /*get predicted locations from existing trackers.*/
        Eigen::MatrixXf trks = Eigen::MatrixXf::Zero(trackers.size(), 5);
        //?TODO: To be deleted trajectories? But it is not judged as NaN later, this array is useless
        std::vector<int> to_del;
        std::vector<Eigen::RowVectorXf> ret;// The result to return? It contains row vectors of shape (1,7)
        // Iterate over trks, row by row
        for (int i = 0; i < trks.rows(); i++) {
            Eigen::RowVectorXf pos = trackers[i].predict();// predict returns a row vector of shape (1,4)
            trks.row(i) << pos(0), pos(1), pos(2), pos(3), 0;
            // note: I don't write the steps to judge if the data is nan, I feel that there will be no nan, so if we don't judge nan, the variable to_del will be useless
        }
        // Calculate speed, shape：(n3,2), used for ORM, the following code simulates python list comprehension
        Eigen::MatrixXf velocities = Eigen::MatrixXf::Zero(trackers.size(), 2);
        Eigen::MatrixXf last_boxes = Eigen::MatrixXf::Zero(trackers.size(), 5);
        Eigen::MatrixXf k_observations = Eigen::MatrixXf::Zero(trackers.size(), 5);
        for (int i = 0; i < trackers.size(); i++) {
            velocities.row(i) = trackers[i].velocity;// Since it's initialized to 0 anyway, there's no need to check if it's None.
            last_boxes.row(i) = trackers[i].last_observation;
            k_observations.row(i) = k_previous_obs(trackers[i].observations, trackers[i].age, delta_t);
        }
        /////////////////////////
        ///  Step1 First round of association
        ////////////////////////
        // Do iou association  associate()
        std::vector<Eigen::Matrix<int, 1, 2>> matched;// Array elements shape is (1,2)
        std::vector<int> unmatched_dets;
        std::vector<int> unmatched_trks;
        auto result = associate(dets_first, trks, iou_threshold, velocities, k_observations, inertia);
        matched = std::get<0>(result);
        unmatched_dets = std::get<1>(result);
        unmatched_trks = std::get<2>(result);
        // Update the matched
        for (auto m: matched) {
            //?TODO The vector used for update is a row vector of shape (1,5), but VectorXf is a column vector of shape (5,1), so an implicit conversion will occur here
            Eigen::Matrix<float, 5, 1> tmp_bbox;
            tmp_bbox = dets_first.block<1, 5>(m(0), 0);
            trackers[m(1)].update(&(tmp_bbox), dets_first(m(0), 5));
        }

        ///////////////////////
        /// Step2 Second round of associaton by OCR to find lost tracks back
        //////////////////////
        // Fusion of BYTE algorithm association
        if (true == use_byte && dets_second.rows() > 0 && unmatched_trks.size() > 0) {
            Eigen::MatrixXf u_trks(unmatched_trks.size(), trks.cols());
            int index_for_u_trks = 0;
            for (auto i: unmatched_trks) {
                u_trks.row(index_for_u_trks++) = trks.row(i);
            }
            Eigen::MatrixXf iou_left = giou_batch(dets_second, u_trks);
            // Replace the function pointer stored in the map
            // Eigen::MatrixXf iou_left = asso_func(dets_second, u_trks);
            if (iou_left.maxCoeff() > iou_threshold) {
                /**
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                 * */
                std::vector<std::vector<float>> iou_matrix(iou_left.rows(), std::vector<float>(iou_left.cols()));
                for (int i = 0; i < iou_left.rows(); i++) {
                    for (int j = 0; j < iou_left.cols(); j++) {
                        iou_matrix[i][j] = -iou_left(i, j);// note：take negative
                    }
                }
                // Linear assignment
                std::vector<int> rowsol, colsol;
                float MIN_cost = execLapjv(iou_matrix, rowsol, colsol, true, 0.01, true);
                std::vector<std::vector<int>> matched_indices;
                // Assign values to matched_indices version :0.1
                for (int i = 0; i < rowsol.size(); i++) {
                    if (rowsol.at(i) >= 0) {
                        matched_indices.push_back({colsol.at(rowsol.at(i)), rowsol.at(i)});
                    }
                }

                std::vector<int> to_remove_trk_indices;
                // Iterate over the results of linear assignment
                for (auto m: matched_indices) {
                    int det_ind = m[0];
                    int trk_ind = unmatched_trks[m[1]];
                    if (iou_left(m[0], m[1]) < iou_threshold) continue;

                    Eigen::Matrix<float, 5, 1> tmp_box;
                    tmp_box = dets_second.block<1, 5>(det_ind, 0);
                    trackers[trk_ind].update(&tmp_box, dets_second(det_ind, 5));
                    to_remove_trk_indices.push_back(trk_ind);
                }
                // Update unmatched_trks
                std::vector<int> tmp_res1(unmatched_trks.size());
                sort(unmatched_trks.begin(), unmatched_trks.end());              // Sort
                sort(to_remove_trk_indices.begin(), to_remove_trk_indices.end());// Sort
                auto end1 = set_difference(unmatched_trks.begin(), unmatched_trks.end(),
                                           to_remove_trk_indices.begin(), to_remove_trk_indices.end(),
                                           tmp_res1.begin());
                tmp_res1.resize(end1 - tmp_res1.begin());
                unmatched_trks = tmp_res1;// Update
            }
        }


        if (unmatched_dets.size() > 0 && unmatched_trks.size() > 0) {
            // Simulate: left_dets = dets[unmatched_dets] unmatched_dets is the detection that has not matched the track
            Eigen::MatrixXf left_dets(unmatched_dets.size(), 6);
            // haha, bug fixed, left_dets.row(inx_for_dets++) = dets_first.row(i) these two indexes are different
            int inx_for_dets = 0;
            for (auto i: unmatched_dets) {
                left_dets.row(inx_for_dets++) = dets_first.row(i);
            }
            // Simulate left_trks = last_boxes[unmatched_trks] unmatched_trks is the track that has not matched the detection
            Eigen::MatrixXf left_trks(unmatched_trks.size(), last_boxes.cols());
            int indx_for_trk = 0;
            for (auto i: unmatched_trks) {
                left_trks.row(indx_for_trk++) = last_boxes.row(i);
            }
            // Calculate cost matrix ?TODO: use iou_batch for now. Do mapping later
            Eigen::MatrixXf iou_left = giou_batch(left_dets, left_trks);
            if (iou_left.maxCoeff() > iou_threshold) {
                /**
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                 * */
                // Find lost tracks
                // note：lapjv uses the library implemented by others
                // First convert iou_left to a two-dimensional vector, and take the negative of the elements during the conversion
                std::vector<std::vector<float>> iou_matrix(iou_left.rows(), std::vector<float>(iou_left.cols()));
                for (int i = 0; i < iou_left.rows(); i++) {
                    for (int j = 0; j < iou_left.cols(); j++) {
                        iou_matrix[i][j] = -iou_left(i, j);// note： take negative
                    }
                }
                // Linear assignment
                std::vector<int> rowsol, colsol;
                float MIN_cost = execLapjv(iou_matrix, rowsol, colsol, true, 0.01, true);
                // Generate rematched_indices
                std::vector<std::vector<int>> rematched_indices;
                // Assign values to rematched_indices version :0.1
                for (int i = 0; i < rowsol.size(); i++) {
                    if (rowsol.at(i) >= 0) {
                        rematched_indices.push_back({colsol.at(rowsol.at(i)), rowsol.at(i)});
                    }
                }
                // If re-linear assignment is not matched, these are all need to be deleted
                std::vector<int> to_remove_det_indices;
                std::vector<int> to_remove_trk_indices;
                // Traverse the results of linear assignment
                for (auto i: rematched_indices) {
                    int det_ind = unmatched_dets[i.at(0)];
                    int trk_ind = unmatched_trks[i.at(1)];
                    if (iou_left(i.at(0), i.at(1)) < iou_threshold) {
                        continue;
                    }
                    ////////////////////////////////
                    ///  Step3  update status of second matched tracks
                    ///////////////////////////////
                    //!fixme: here update, because it is re-matched
                    Eigen::Matrix<float, 5, 1> tmp_bbox;
                    tmp_bbox = dets_first.block<1, 5>(det_ind, 0);
                    trackers.at(trk_ind).update(&tmp_bbox, dets_first(det_ind, 5));
                    to_remove_det_indices.push_back(det_ind);
                    to_remove_trk_indices.push_back(trk_ind);
                }
                // Update unmatched_dets & trks, simulate setdiff1d function
                std::vector<int> tmp_res(unmatched_dets.size());
                // note: because set_difference requires the data to be sorted before comparison, sorting these data should not cause any bugs
                sort(unmatched_dets.begin(), unmatched_dets.end());              // Sort
                sort(to_remove_det_indices.begin(), to_remove_det_indices.end());// Sort
                auto end = set_difference(unmatched_dets.begin(), unmatched_dets.end(),
                                          to_remove_det_indices.begin(), to_remove_det_indices.end(),
                                          tmp_res.begin());
                tmp_res.resize(end - tmp_res.begin());
                unmatched_dets = tmp_res;// Update
                std::vector<int> tmp_res1(unmatched_trks.size());
                sort(unmatched_trks.begin(), unmatched_trks.end());              // Sort
                sort(to_remove_trk_indices.begin(), to_remove_trk_indices.end());// Sort
                auto end1 = set_difference(unmatched_trks.begin(), unmatched_trks.end(),
                                           to_remove_trk_indices.begin(), to_remove_trk_indices.end(),
                                           tmp_res1.begin());
                tmp_res1.resize(end1 - tmp_res1.begin());
                unmatched_trks = tmp_res1;// Update
            }
        }

        //!fixme: As mentioned in the paper, if the existing tracks do not have observations, they are also updated
        for (auto m: unmatched_trks) {
            // python version gives update z=None, but in C++ version, we can just give nullptr
            trackers.at(m).update(nullptr, 0);
        }
        ///////////////////////////////
        /// Step4 Initialize new tracks and remove expired tracks
        ///////////////////////////////
        /*create and initialise new trackers for unmatched detections*/
        for (int i: unmatched_dets) {
            Eigen::RowVectorXf tmp_bbox = dets_first.block(i, 0, 1, 5);
            // dets_first(i, 5) is the class of the target
            int cls_ = int(dets_first(i, 5));
            KalmanBoxTracker trk = KalmanBoxTracker(tmp_bbox, cls_, delta_t);
            // Append newly created to trackers
            trackers.push_back(trk);
        }
        int tmp_i = trackers.size();// !fixme: not sure what this is for, seems to be used to save MOT format test results
                                    // Reverse traverse trackers array to generate results to return
        for (int i = trackers.size() - 1; i >= 0; i--) {
            // Get prediction values, there are two ways, the difference is not that big
            Eigen::Matrix<float, 1, 4> d;
            int last_observation_sum = trackers.at(i).last_observation.sum();
            if (last_observation_sum < 0) {
                d = trackers.at(i).get_state();
            } else {
                /**
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                 * */
                d = trackers.at(i).last_observation.block(0, 0, 1, 4);
            }
            /**
                If no object is detected within a certain threshold (usually 1 frame), the current tracker is marked as "not updated".

The condition `time_since_update < 1` means that the tracker correctly matched the target in the previous frame and predicted its position in the current frame,

and the position in the current frame has not exceeded a threshold (1 frame). This indicates that the tracker is still effective and can be added to `ret` as a matching result.
             * */
            if (trackers.at(i).time_since_update < 1 && ((trackers.at(i).hit_streak >= min_hits) | (frame_count <= min_hits))) {
                // +1 as MOT benchmark requires positive
                // d is coordinates 1x4 , trackers.at(i).id ,cls,conf are all 1x1 scalars
                // Combine them to form a 1x7 row vector
                Eigen::RowVectorXf tracking_res(7);
                // note: ID starts from 1, Conforms to MOT format
                tracking_res << d(0), d(1), d(2), d(3), trackers.at(i).id + 1, trackers.at(i).cls, trackers.at(i).conf;
                ret.push_back(tracking_res);
            }
            // remove dead tracklets
            if (trackers.at(i).time_since_update > max_age) {
                /*here need to delete the element at the specified position*/
                trackers.erase(trackers.begin() + i);
            }
        }
        return ret;
    }
}// namespace ocsort