#ifndef OC_SORT_CPP_KALMANBOXTRACKER_H
#define OC_SORT_CPP_KALMANBOXTRACKER_H
////todo: The pointer part will be optimized into smart pointers later.///
////////////// KalmanBoxTracker /////////////
#include "../include/Utilities.h"
#include "../include/kalmanfilter.h"
#include "iostream"
namespace ocsort {

    class KalmanBoxTracker {
    public:
        /*method*/
        KalmanBoxTracker() : kf(nullptr) {};
        ~KalmanBoxTracker() { delete kf; }// Free allocated memory
        KalmanBoxTracker(Eigen::VectorXf bbox_, int cls_, int delta_t_ = 3);
        void update(Eigen::Matrix<float, 5, 1> *bbox_, int cls_);
        Eigen::RowVectorXf predict();// Returns a (1,4) row vector
        Eigen::VectorXf get_state(); // Returns state vector x (1,4)

    public:
        /*variable*/
        static int count;
        Eigen::VectorXf bbox;// [5,1]
        //? TODO: replace to smart pointer
        KalmanFilterNew *kf;// kalman predictor
        int time_since_update;
        int id;
        std::vector<Eigen::VectorXf> history;//  [4,1]
        int hits;
        int hit_streak;
        int age = 0;
        float conf;
        int cls;
        // Below variables are newly added in ocsort
        Eigen::RowVectorXf last_observation = Eigen::RowVectorXf::Zero(5);
        std::map<int, Eigen::VectorXf> observations;
        std::vector<Eigen::VectorXf> history_observations;
        Eigen::RowVectorXf velocity = Eigen::RowVectorXf::Zero(2);// [2,1]
        int delta_t;
    };
}// namespace ocsort

#endif//OC_SORT_CPP_KALMANBOXTRACKER_H
