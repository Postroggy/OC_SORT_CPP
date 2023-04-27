#include "OCsort.h"
#include "iostream"
#include "vector"
#include <Eigen/Core>
#include "numeric"
using namespace std;
using namespace Eigen;
using namespace ocsort;
using ocsort::associate;
std::vector<int> create_vector_from_0_to_n(int n) {
    std::vector<int> result(n);
    std::iota(result.begin(), result.end(), 0);
    return result;
}
int main(int argc, char *argv[]) {
    Eigen::MatrixXd dets(2, 6), tracks(2, 5), valocity = MatrixXd(2, 2);
    valocity << 0, 0, 0, 0;
    Eigen::MatrixXd prev_ob(2, 5);
    prev_ob.fill(-1);
    float vdc = 0.3941737016672115;
    dets << 851., 203., 923., 364., 0.69, 0., 0., 153., 51., 451., 0.52, 0.;
    OCSort A = OCSort(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, false);
    // 测试 update 函数
    A.update(dets);

}
