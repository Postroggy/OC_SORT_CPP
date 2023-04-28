#include "OCsort.h"
#include "iostream"
#include "numeric"
#include "vector"
#include <Eigen/Core>
using namespace std;
using namespace Eigen;
using namespace ocsort;
using ocsort::associate;
template<typename AnyCls>
ostream &operator<<(ostream &os, const vector<AnyCls> &v) {
    os << "{";
    for (auto it = v.begin(); it != v.end(); ++it) {
        os << "(" << *it << ")";
        if (it != v.end() - 1) os << ", ";
    }
    os << "}";
    return os;
}
int main(int argc, char *argv[]) {
    Eigen::MatrixXd tracks(2, 5), valocity = MatrixXd(2, 2);
    Eigen::MatrixXd dets1(2, 6), dets2(2, 6), dets3(2, 6);
    valocity << 0, 0, 0, 0;
    Eigen::MatrixXd prev_ob(2, 5);
    prev_ob.fill(-1);
    float vdc = 0.3941737016672115;
    dets1 << 851., 203., 923., 364., 0.69, 0.,
            0., 153., 51., 451., 0.52, 0.;
    /*res:
[[  0.   153.    51.   451.     2.     0.     0.52]
[851.   203.   923.   364.     1.     0.     0.69]]
     * */
    dets2 << 0., 145., 55., 454., 0.66, 0.,
            849., 205., 924., 364., 0.63, 0.;
    /*
     * res:
[[  0.   145.    55.   454.     2.     0.     0.66]
[849.   205.   924.   364.     1.     0.     0.63]]*/
    dets3 << 0., 144., 59., 454., 0.67, 0.,
            849., 198., 921., 365., 0.59, 0.;
    /*res:
[[  0.   144.    59.   454.     2.     0.     0.67]
[849.   198.   921.   365.     1.     0.     0.59]]*/
    OCSort A = OCSort(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, false);
    /* 测试 update 函数,*/
    auto res = A.update(dets1);
    cout << "First: " << res << endl;
    // todo: 第二个检测出问题了，返回的结果为空了
    res = A.update(dets2);
    cout << "Second: " << res << endl;
    res = A.update(dets3);
    cout << "Third: " << res << endl;
}
