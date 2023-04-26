#include "Eigen/Dense"
#include "iostream"
using namespace std;
using namespace Eigen;

int main(int argc, char *argv[]) {
    Eigen::MatrixXd dets(2, 4), tracks(2, 4);
    dets << 2, 1, 7, 8,
            6, 2, 2, 2;
    tracks << 1, 6, 5, 7,
            2, 8, 1, 5;
}