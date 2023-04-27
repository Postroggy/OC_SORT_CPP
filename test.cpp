#include "../include/association.h"
#include "iostream"
#include <Eigen/Core>
#include "vector"
using namespace std;
using namespace Eigen;
using ocsort::associate;

int main(int argc, char *argv[]) {
    Eigen::MatrixXd dets(2, 6), tracks(2, 5), valocity = MatrixXd(2, 2);
    valocity << 0, 0, 0, 0;
    Eigen::MatrixXd prev_ob(2, 5);
    prev_ob.fill(-1);
    float vdc = 0.3941737016672115;
    dets << 0., 145., 55., 454., 0.66, 0., 849., 205., 924., 364., 0.63, 0.;
    tracks << 851., 203., 923., 364., 0., 0., 153., 51., 451., 0.;
    valocity << 1, 2, 3, 4;
    auto result = associate(dets, tracks, 0.1, valocity, prev_ob, vdc);
    auto matches = std::get<0>(result);
    auto unmatched_detections = std::get<1>(result);
    auto unmatched_trackers = std::get<2>(result);
}
