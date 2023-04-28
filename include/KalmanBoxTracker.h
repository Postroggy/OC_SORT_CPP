#ifndef OC_SORT_CPP_KALMANBOXTRACKER_H
#define OC_SORT_CPP_KALMANBOXTRACKER_H
////todo: 指针这部分后续优化成智能指针///
////////////// KalmanBoxTracker /////////////
#include "../include/kalmanfilter.h"
#include "../include/Utilities.h"
#include "iostream"
/*
This class represents the internal state of individual
tracked objects observed as bbox.
*/
namespace ocsort {

    class KalmanBoxTracker {
    public:
        /*method*/
        KalmanBoxTracker(){};
        KalmanBoxTracker(Eigen::VectorXd bbox_, int cls_, int delta_t_ = 3);
        void update(Eigen::VectorXd *bbox_, int cls_);
        Eigen::VectorXd predict();  // 返回的是 (1,4)的行向量
        Eigen::VectorXd get_state();// 返回状态向量 x (1,4)

    public:
        /*variable*/
        static int count;
        Eigen::VectorXd bbox;// [5,1]
        // why? 为什么必须要指针才行？
        KalmanFilterNew *kf;// kalman预测器
        int time_since_update;
        int id;
        std::vector<Eigen::VectorXd> history;//  [4,1]的吧？
        int hits;
        int hit_streak;
        int age = 0;
        double conf;
        int cls;
        // 下面变量是 ocsort 相较于 sort 增加的, fixme: 实际上这里应该是(1,5)的呀？ python 版本是 (1,5),这里是行向量看会不会出问题
        Eigen::RowVectorXd last_observation = Eigen::RowVectorXd::Zero(5);
        std::unordered_map<int, Eigen::VectorXd> observations;// 这里必须是hash类型的，才能跟python的dict()对应
        std::vector<Eigen::VectorXd> history_observations;
        Eigen::RowVectorXd velocity = Eigen::RowVectorXd::Zero(2);// [2,1]
        int delta_t;
    };
}// namespace ocsort

#endif//OC_SORT_CPP_KALMANBOXTRACKER_H
