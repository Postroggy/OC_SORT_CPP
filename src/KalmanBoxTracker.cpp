#include "../include/KalmanBoxTracker.h"
#include <utility>

namespace ocsort {
    // Used for assigning IDs, just increment them.
    int KalmanBoxTracker::count = 0;
    KalmanBoxTracker::KalmanBoxTracker(Eigen::VectorXf bbox_, int cls_, int delta_t_) {
        // note: Input: bbox: should be 1x5, currently it is 5x1, cls: integer, delta_t: integer,
        bbox = std::move(bbox_);// convert to z
        delta_t = delta_t_;
        // Initialize kalman filter (new)//
        // kf = KalmanFilterNew(7,4);
        // auto a = KalmanFilterNew(7,4);
        // kf = KalmanFilterNew(7,4);
        kf = new KalmanFilterNew(7, 4);
        kf->F << 1, 0, 0, 0, 1, 0, 0,
                0, 1, 0, 0, 0, 1, 0,
                0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 1;
        kf->H << 1, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
        kf->R.block(2, 2, 2, 2) *= 10.0;
        kf->P.block(4, 4, 3, 3) *= 1000.0;
        kf->P *= 10.0;
        kf->Q.bottomRightCorner(1, 1)(0, 0) *= 0.01;
        kf->Q.block(4, 4, 3, 3) *= 0.01;
        // Assign the first 4 rows of x[7,1] to the result of convert_bbox_to_z(bbox)
        kf->x.head<4>() = convert_bbox_to_z(bbox);
        time_since_update = 0;
        id = KalmanBoxTracker::count;
        KalmanBoxTracker::count += 1;
        history.clear();// Empty array
        hits = 0;       // Number of matches
        hit_streak = 0; // Number of consecutive matches
        age = 0;        // Number of frames since the object started being tracked
        conf = bbox(4); // index from 0, the last one is confidence
        cls = cls_;
        ////////////////////////////////////////////////////
        // NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        // function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        // fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        // let's bear it for now.
        //////////////////////////////////////////////////////
        last_observation.fill(-1);   // Placeholder [-1,-1,-1,-1,-1]
        observations.clear();        // Type: map<int, Eigen::VectorXf>
        history_observations.clear();// Type: std::vector<Eigen::VectorXf>
        velocity.fill(0);            // Type: Eigen::VectorXf [2,1]
    }
    /**
     * Updates the state vector with observed bbox.
     * @param bbox_
     * @param cls_ This variable is not present in the original SORT
     */
    void KalmanBoxTracker::update(Eigen::Matrix<float, 5, 1> *bbox_, int cls_) {
        if (bbox_ != nullptr) {
            // Observation matched
            conf = (*bbox_)[4];
            cls = cls_;
            if (int(last_observation.sum()) >= 0) {
                Eigen::VectorXf previous_box_tmp;
                for (int dt = delta_t; dt > 0; --dt) {
                    auto it = observations.find(age - dt);
                    if (it != observations.end()) {
                        previous_box_tmp = it->second;
                        break;
                    }
                }
                if (0 == previous_box_tmp.size()) {     // If previous_box_tmp is not assigned in the previous for-loop
                    previous_box_tmp = last_observation;// Then assign the last observation to it
                }
                // small patch to improve performance
                // remove redundant old data, NOTE: it may cause malfunction on tracking accuracy
                const int maxSize = 300;
                if (observations.size() > maxSize) {
                    observations.erase(observations.begin());
                }
                ////////////////////////
                //// Estimate the track speed direction with observations \Delta t steps away//
                ////////////////////////
                /* velocity is (2,1), previous_box_tmp is (5,1), bbox is (5,1) */
                velocity = speed_direction(previous_box_tmp, *bbox_);
            }
            ///////////////////////
            //// Insert new observations. This is a ugly way to maintain both self.observations
            //// and self.history_observations. Bear it for the moment.
            //////////////////////
            // ocsort new
            last_observation = *bbox_; // last_observation is 1x5 row vector
            observations[age] = *bbox_;// This is 5x1 column vector
            history_observations.push_back(*bbox_);
            // traditional sort
            time_since_update = 0;
            history.clear();
            hits += 1;
            hit_streak += 1;
            Eigen::VectorXf tmp = convert_bbox_to_z(*bbox_);
            kf->update(tmp);
        } else {
            /* If there is no detection, also update, the KalmanFilter function is prepared to handle this case */
            kf->update(Eigen::VectorXf());
        }
    }

    /**
     * Returns a (1,4) row vector
     */
    Eigen::RowVectorXf KalmanBoxTracker::predict() {
        ///////////////////////
        //// Advances the state vector and returns the predicted bounding box estimate.
        //////////////////////
        if (kf->x[6] + kf->x[2] <= 0) kf->x[6] *= 0.0;
        kf->predict();
        age += 1;
        if (time_since_update > 0) hit_streak = 0;
        time_since_update += 1;
        // fixme: here append to history should be kf->x instead of kf->z
        auto vec_out = convert_x_to_bbox(kf->x);
        history.push_back(vec_out);
        return vec_out;
    }
    Eigen::VectorXf KalmanBoxTracker::get_state() {
        return convert_x_to_bbox(kf->x);
    }
}// namespace ocsort