//#include "../include/KalmanBoxTracker.h"
#include "../include/OCsort.h"
int main() {
    auto a = ocsort::OCSort();
    // bbox 是 [5,1]
    Eigen::MatrixXd bbox = Eigen::MatrixXd::Random(2, 6);
//    std::cout << bbox << std::endl;
//    auto A = ocsort::OCSort(0.6);
//    A.update(bbox);
    //    Eigen::MatrixXd bboxes1(2, 4);
//    bboxes1 << 3, 3, 7, 6,
//            2, 1, 5, 4;
//    Eigen::MatrixXd bboxes2(3, 4);
//    bboxes2 << 2, 2, 6, 5,
//            1, 0, 4, 3,
//            5, 4, 8, 7;
//    auto a =ocsort::iou_batch(bboxes1,bboxes2);
//    std::cout<<a<<std::endl;
}
