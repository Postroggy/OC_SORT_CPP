#include "OCsort.h"
#include "iostream"
#include <Eigen/Core>
#include "vector"
using namespace std;
using namespace Eigen;
using namespace ocsort;
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
    OCSort A = OCSort(0,50,1,0.22136877277096445,1,"giou",0.3941737016672115,false);

}
