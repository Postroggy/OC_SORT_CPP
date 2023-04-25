#ifndef OC_SORT_CPP_KALMANBOXTRACKER_H
#define OC_SORT_CPP_KALMANBOXTRACKER_H
#pragma once
////todo: 指针这部分后续优化成智能指针///
////////////// KalmanBoxTracker /////////////
#include "../include/kalmanfilter.h"
#include "Eigen/Dense"
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
//        void update(Eigen::VectorXd* bbox_, int cls_);

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
        float conf;
        int cls;
        // 下面变量是 ocsort 相较于 sort 增加的
        Eigen::VectorXd last_observation;
        std::map<int, Eigen::VectorXd> observations;
        std::vector<Eigen::VectorXd> history_observations;
        Eigen::VectorXd velocity;// [2,1]
        int delta_t;
    };
}// namespace ocsort

#endif//OC_SORT_CPP_KALMANBOXTRACKER_H
