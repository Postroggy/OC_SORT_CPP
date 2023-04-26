#include "Eigen/Dense"
#include "iostream"
using namespace std;
using namespace Eigen;
#include <Eigen/Core>

#include <Eigen/Core>



int main(int argc, char *argv[]) {
    Eigen::MatrixXd dets(2, 4), tracks(2, 4);
    dets << 2, 1, 7, 8,
            6, 2, 2, 2;
    tracks << 1, 6, 5, 7,
            2, 8, 1, 5;
    MatrixXd inertia_X(2, 2);
    MatrixXd inertia_Y(2, 2);
    MatrixXd X(2, 2);
    MatrixXd Y(2, 2);
    X << 0.59999976, 0.21693041, 0.83205006, 0.48564284;
    Y << -0.79999968, -0.97618685, -0.55470004, -0.87415711;
    inertia_X << -0.44721346, -0.44721346, 0.97014203, 0.97014203;
    inertia_Y << -0.89442692, -0.89442692, -0.24253551, -0.24253551;
    MatrixXd diff_angle_cos(2, 2);
    diff_angle_cos << 0.44721328, 0.7761136, 0.9417412, 0.68315667;
    MatrixXd diff_angle = Eigen::acos(diff_angle_cos.array());


    Eigen::MatrixXd previous_obs(5, 5);
    previous_obs << 1, 2, 3, 4, -1,
            5, 6, 7, -1, -2,
            9, 10, 11, 12, 13,
            14, -3, 15, 16, 17,
            18, 19, 20, -4, 21;
    Eigen::Array<bool, Eigen::Dynamic, 1> valid_mask = Eigen::Array<bool, Eigen::Dynamic, 1>::Ones(previous_obs.rows());
    valid_mask.colwise() *= (previous_obs.col(4).array() >= 0);

    Eigen::MatrixXd a(2, 6);
    Eigen::MatrixXd b(2, 5);
    a << 0., 145., 55., 454., 0.66, 0.,
            849., 205., 924., 364., 0.63, 0.;
    b << 851., 203., 923., 364., 0.,
            0., 153., 51., 451., 0.;
    cout<<    iou_batch(a, b);

}