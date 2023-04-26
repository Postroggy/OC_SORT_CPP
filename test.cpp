#include "Eigen/Dense"
#include "iostream"
using namespace std;
using namespace Eigen;
#include "../include/association.h"
#include <Eigen/Core>


int main(int argc, char *argv[]) {
    Eigen::MatrixXd dets(2, 5), tracks(2, 5),valocity = MatrixXd::Zero(2,2);
    Eigen::MatrixXd prev_ob(2,5);
    float vdc = 0.3941737016672115;
    prev_ob.fill(-1);
    dets << 0., 145., 55., 454., 0.66, 0., 849., 205., 924., 364., 0.63, 0.;
    tracks << 851., 203., 923., 364., 0., 0., 153., 51., 451., 0.;
    ocsort::associate(dets,tracks,0.1,valocity,prev_ob,vdc);
    // todo: 4月26日 23：11 写到这里，吃泡面先
}