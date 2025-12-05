#ifndef OC_SORT_CPP_OCSORT_H
#define OC_SORT_CPP_OCSORT_H
#include "KalmanBoxTracker.h"
#include "association.h"
#include "lapjv.h"
#include <functional>
#include <unordered_map>
namespace ocsort {

    class OCSort {
    public:
        OCSort(float det_thresh_, int max_age_ = 30, int min_hits_ = 3, float iou_threshold_ = 0.3,
               int delta_t_ = 3, std::string asso_func_ = "iou", float inertia_ = 0.2, bool use_byte_ = false);
        std::vector<Eigen::RowVectorXf> update(Eigen::MatrixXf dets);// Input is [n1,6], returns matrix [n2,7]. When tracking result is empty, returns all zeros (why does it return (0,5) shape then?)

    public:
        float det_thresh;
        int max_age;
        int min_hits;
        float iou_threshold;
        int delta_t;
        std::function<Eigen::MatrixXf(const Eigen::MatrixXf &, const Eigen::MatrixXf &)> asso_func;
        float inertia;
        bool use_byte;
        // Used to store KalmanBoxTracker
        std::vector<KalmanBoxTracker> trackers;
        int frame_count;
    };

}// namespace ocsort
#endif//OC_SORT_CPP_OCSORT_H
