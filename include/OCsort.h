#ifndef OC_SORT_CPP_OCSORT_H
#define OC_SORT_CPP_OCSORT_H
#include "KalmanBoxTracker.h"
#include "association.h"
#include "lapjv.h"
namespace ocsort {

    class OCSort {
    public:
        OCSort(float det_thresh_, int max_age_ = 30, int min_hits_ = 3, float iou_threshold_ = 0.3,
               int delta_t_ = 3, std::string asso_func_ = "iou", float inertia_ = 0.2, bool use_byte_ = false);
        std::vector<Eigen::RowVectorXd> update(Eigen::MatrixXd dets);//输入是[n1,6]的大小， 返回的矩阵是 [n2,7]。当追踪结果为空时，返回全0(此时为啥他返回(0,5)的shape？)

    public:
        float det_thresh;
        int max_age;
        int min_hits;
        float iou_threshold;
        int delta_t;
        std::string asso_func;
        float inertia;
        bool use_byte;
        // 用来保存 KalmanBoxTracker
        std::vector<KalmanBoxTracker> trackers;
        int frame_count;
    };

}// namespace ocsort
#endif//OC_SORT_CPP_OCSORT_H
