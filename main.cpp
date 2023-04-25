#include "../include/KalmanBoxTracker.h"
#include <iostream>
int main() {
    // bbox 是 [5,1]
    Eigen::VectorXd bbox = Eigen::VectorXd::Ones(5, 1);
    auto A = ocsort::KalmanBoxTracker(bbox, 0, 3);
    std::cout << A.kf->Q << std::endl;
}