#include <utility>

#include "../include/KalmanBoxTracker.h"
namespace ocsort {
    // 用于分配ID的，递增就行
    int KalmanBoxTracker::count = 0;
    KalmanBoxTracker::KalmanBoxTracker(Eigen::VectorXd bbox_, int cls_, int delta_t_) {
        // note： 输入： bbox: 应为1x5，目前是 5x1, cls:整形，delta_t：整形,
        bbox = std::move(bbox_);// 还要 convert to z
        delta_t = delta_t_;
        //初始化kalman filter (new)//
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
        // 下面是要赋值给x[7,1]，但是bbox是[5,1] convert_bbox_to_z(bbox)是(4,1)， 前4行赋值给x
        kf->x.head<4>() = convert_bbox_to_z(bbox);
        time_since_update = 0;
        id = KalmanBoxTracker::count;
        KalmanBoxTracker::count += 1;
        history.clear();// 空的数组
        hits = 0;       // 匹配上的次数
        hit_streak = 0; // 连续匹配上的次数
        age = 0;        // 示自该物体开始跟踪以来已经过去的帧数
        conf = bbox(4); // index从0开始，最后一个是置信度
        cls = cls_;
        ////////////////////////////////////////////////////
        // NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        // function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        // fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        // let's bear it for now.
        //////////////////////////////////////////////////////
        last_observation.fill(-1);   // 占位符 [-1,-1,-1,-1,-1]
        observations.clear();        // 类型：map<int, Eigen::VectorXd>
        history_observations.clear();// 类型：std::vector<Eigen::VectorXd>
        velocity.fill(0);            // 类型：Eigen::VectorXd [2,1]
    }
    /**
     * Updates the state vector with observed bbox.
     * @param bbox_
     * @param cls_ 这个变量在原sort中是没有的
     */
    void KalmanBoxTracker::update(Eigen::Matrix<double, 5, 1> *bbox_, int cls_) {
        // todo :WARNING:

//        std::cout << "============ENTER KalmanBOXTracker::update==========\n";
        if (bbox_ != nullptr) {
            // note: 调试用
//            std::cout << "=======Congrats it's not empty=======\n";
            // 匹配到观测值了
            conf = (*bbox_)[4];
            cls = cls_;
            // note: 调试用
//            std::cout << "conf\n"
//                      << conf << "\ncls\n"
//                      << cls << "\nlast_observation.sum()\n"
//                      << last_observation.sum() << std::endl;
            // fixme:完全没看懂，有 last_observation 则没有 previous_observation 吗？
            if (int(last_observation.sum()) >= 0) {
                // 00:32 写到这，先睡觉了
                Eigen::VectorXd previous_box_tmp;
                for (int i = 0; i < delta_t; ++i) {
                    int dt = delta_t - i;
                    if (observations.count(age - dt) > 0) {
                        // 如果在map中存在
                        previous_box_tmp = observations[age - dt];
                        break;// 跳出循环
                    }
                }
                if (0 == previous_box_tmp.size()) {     // 如果previous_box_tmp并没有在上一个for-loop中被赋值
                    previous_box_tmp = last_observation;// 则将上一个观测值赋给他
                }
                //note: 调试用
//                std::cout << "last_observation.sum() is less than 0, previous_box_tmp is" << previous_box_tmp << std::endl;
                ////////////////////////
                //// Estimate the track speed direction with observations \Delta t steps away//
                ////////////////////////
                /* velocity是(2,1)的，previous_box_tmp是(5,1) bbox是(5,1) */
                velocity = speed_direction(previous_box_tmp, *bbox_);
            }
            ///////////////////////
            //// Insert new observations. This is a ugly way to maintain both self.observations
            //// and self.history_observations. Bear it for the moment.
            //////////////////////
            // ocsort 新增的
            // todo:这个 last_observation 赋值好像有问题，没事了，我发现注释只能用单行的，msvc编译器有问题
            last_observation = *bbox_; // last_observation 是 1x5 行向量
            observations[age] = *bbox_;// 这里保存的是 5x1 的列向量
            history_observations.push_back(*bbox_);
            //            std::cout << "WARING: last_observation: \n"
            //                      << last_observation << std::endl;
            // sort通用代码
            time_since_update = 0;
            history.clear();// 空
            hits += 1;
            hit_streak += 1;
            Eigen::VectorXd tmp = convert_bbox_to_z(*bbox_);
            kf->update(&tmp);// todo 这里可能有bug
        } else {
//            std::cout << "What a pity bbox used in update is nullptr\n";
            /*如果没有检测到 bbox，也更新，KalmanFilter函数写好了应对这种情况的*/
            kf->update(nullptr);
        }
        // note: 调试用
//        std::cout << "============EXITING KalmanBOXTracker::update==========\n";
    }

    /**
     * 这里返回的是 (1,4) 的行向量
     */
    Eigen::RowVectorXd KalmanBoxTracker::predict() {
        ///////////////////////
        //// Advances the state vector and returns the predicted bounding box estimate.
        //////////////////////
        // note: 调试用
//        std::cout << "Now you are in KalmanBoxTracker::predict func\n";
        if (kf->x[6] + kf->x[2] <= 0) kf->x[6] *= 0.0;
        // note: 调试用
        //        std::cout<<"before predict kf=>x:\n"<<kf->x<<std::endl;
        kf->predict();
        // note: 调试用
        //        std::cout<<"after predict kf=>x:\n"<<kf->x<<std::endl;
        age += 1;
        if (time_since_update > 0) hit_streak = 0;
        time_since_update += 1;
        // fixme: 发现自己写错了，这里 append刀 history的应该是 kf->x 而不是 kf->z
        history.push_back(convert_x_to_bbox(kf->x));
        return convert_x_to_bbox(kf->x);// 返回 history 中的最后一个元素
    }
    Eigen::VectorXd KalmanBoxTracker::get_state() {
        // todo： 这个函数还是有点问题啊，有问题，一个变量  w 写成 h了，改过来了
        //        std::cout<<"before convert:\n"<<kf->x<<std::endl;
        //        std::cout<<"after convert:\n"<<convert_x_to_bbox(kf->x)<<std::endl;
        return convert_x_to_bbox(kf->x);
    }
}// namespace ocsort