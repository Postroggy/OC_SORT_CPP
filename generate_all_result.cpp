#include <Eigen/Dense>
#include <OCsort.h>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
using namespace Eigen;
using namespace ocsort;
Eigen::Matrix<double, Eigen::Dynamic, 6> read_csv_to_eigen(const std::string &filename) {
    // 读取CSV文件
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
        exit(1);
    }
    // 解析CSV格式
    std::string line;
    std::vector<std::vector<float>> data;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::istringstream iss(line);
        std::string field;
        while (std::getline(iss, field, ',')) {
            row.push_back(std::stof(field));
        }
        data.push_back(row);
    }
    // 转换为Eigen::Matrix
    Eigen::Matrix<double, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}

template<typename AnyCls>
// 重载 << 输出 vector
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
    // 初始化 OCSort 对象
    OCSort A = OCSort(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, false);
    // 读取数据
    std::ostringstream filename;
    for (int i = 1; i < 451; i++) {
        // 读取输入数据
        std::cout << "============== " << i << " =============" << std::endl;
        filename << "../../figure_ocsort/MOT_xyxys/private/SeedDet/MOT17-01/" << i << ".csv";
        Eigen::Matrix<double, Eigen::Dynamic, 6> dets = read_csv_to_eigen(filename.str());
        filename.str("");
        // 推理
        std::vector<Eigen::RowVectorXd> res = A.update(dets);
        // 打印输出
        std::cout << "predict:\n"
                  << res << std::endl;
        // 保存输出
        ofstream file;
        filename << "../OUTPUT_DATA/MOT17-01/" << i << ".csv";
        file.open(filename.str());
        filename.str("");
        IOFormat CSVFormat(StreamPrecision, DontAlignCols, ",", "\n");
        for (auto i: res) {
            file << i.format(CSVFormat) << endl;
        }
        file.close();
    }
    return 0;
}