#include <utility>

#include "../include/KalmanBoxTracker.h"
#include "../include/Utilities.h"
#include "Eigen/Dense"
namespace ocsort {
    // 用于分配ID的，递增就行
    int KalmanBoxTracker::count = 0;
    KalmanBoxTracker::KalmanBoxTracker(Eigen::VectorXd bbox_, int cls_, int delta_t_) {
        bbox = std::move(bbox_);
        delta_t = delta_t_;
        //初始化kalman filter (new)//
        //        kf = KalmanFilterNew(7,4);
        //        auto a = KalmanFilterNew(7,4);
        //        kf = KalmanFilterNew(7,4);
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
        // 下面是要赋值给x[7,1]，但是bbox是[5,1] convert_bbox_to_z(bbox)是(4,1)
        kf->x.head<4>() = convert_bbox_to_z(bbox);
        time_since_update = 0;
        id = KalmanBoxTracker::count;
        KalmanBoxTracker::count += 1;
        history;       // 空的数组
        hits = 0;      // 匹配上的次数
        hit_streak = 0;// 连续匹配上的次数
        age = 0;       // 示自该物体开始跟踪以来已经过去的帧数
        conf = (float) bbox[5];
//        cls = cls_;
//        ////////////////////////////////////////////////////
//        // NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
//        // function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
//        // fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
//        // let's bear it for now.
//        //////////////////////////////////////////////////////
//        last_observation << -1,-1,-1,-1,-1; // 占位符
//        observations; // 类型：map<int, Eigen::VectorXd>
//        history_observations; // 类型：std::vector<Eigen::VectorXd>
//        velocity; // 类型：Eigen::VectorXd [2,1]

    }
    /**
     * Updates the state vector with observed bbox.
     * @param bbox_
     * @param cls_ 这个变量在原sort中是没有的
     */
//    void KalmanBoxTracker::update(Eigen::VectorXd* bbox_, int cls_) {
//
//    }
}// namespace ocsort