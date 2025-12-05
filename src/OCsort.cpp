#include "../include/OCsort.h"
#include "iomanip"
#include <utility>

namespace ocsort {
    /*Overload << for vector, can print vector directly*/
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
        // Below added in ocsort
        det_thresh = det_thresh_;
        delta_t = delta_t_;
        // Declare unordered_map, key is string, value is function object of function pointer type with no args and no return value (actually takes args and returns MatrixXf)
        std::unordered_map<std::string, std::function<Eigen::MatrixXf(const Eigen::MatrixXf &, const Eigen::MatrixXf &)>> ASSO_FUNCS{
                {"iou", iou_batch},
                {"giou", giou_batch}};
        ;
        // Determine function to use later, although function pointer is saved, actually I didn't use it
        std::function<Eigen::MatrixXf(const Eigen::MatrixXf &, const Eigen::MatrixXf &)> asso_func = ASSO_FUNCS[asso_func_];
        // asso_func = std::move(asso_func_);// todo Should I implement this with function pointer?
        inertia = inertia_;
        use_byte = use_byte_;
        KalmanBoxTracker::count = 0;
    }
    // fixme: This controls print precision, remove when releasing
    std::ostream &precision(std::ostream &os) {
        os << std::fixed << std::setprecision(2);
        return os;
    }
    std::vector<Eigen::RowVectorXf> OCSort::update(Eigen::MatrixXf dets) {
        /*
            if (trackers.at(i).time_since_update > max_age) {
                /*Delete element at specified position*/
        trackers.erase(trackers.begin() + i);
    }
}
return ret;
}
}// namespace ocsort