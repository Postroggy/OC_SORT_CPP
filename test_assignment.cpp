#include "Eigen/Dense"
#include "iostream"
#include "lapjv.h"
#include "vector"
using namespace std;
using namespace Eigen;

int main(int argc, char *argv[]) {
    // 测试线性分配函数
    vector<vector<double>> cost_matrix = {{-0.16, 0., -0.22, 0., -0.24, 0., 0.},
                                         {0., 0., 0., 0., 0., 0., 0.},
                                         {0., 0., 0., -0.04, 0., 0., 0.}};
    // expected output：
    /*
            x:  [4 1 3]
            y:  [-1  1 -1  2  0 -1 -1]
            remain_inds [[0 4]
                        [1 1]
                        [2 3]]
     * */
    vector<int> rowsol, colsol;
    double cost = execLapjv(cost_matrix, rowsol, colsol, true, 0.01, true);
    cout << "Minimum Cost: " << cost << endl;
    cout << "\nPrint Result\n";
    for (auto i: rowsol) {
        cout << " " << i;
    }
    cout << "\n";
    for (auto i: rowsol) {
        cout << " " << i;
    }
}