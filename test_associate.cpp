#include "OCsort.h"
#include "fstream"
#include "iostream"
#include "numeric"
#include "vector"
#include <Eigen/Core>
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
    //    Matrix<double, 3, 6> dets_first;
    //    dets_first << 0.00, 120.00, 81.00, 451.00, 0.82, 0.00,
    //            847.00, 194.00, 913.00, 370.00, 0.70, 0.00,
    //            652.00, 226.00, 692.00, 328.00, 0.54, 0.00;
    //    // todo :WARNING: 这个预测的 tracker 和py版本的有偏差，我去，差了几个像素点了。
    //    // todo :WARNING: 也就是说我的 KalmanBoxTracker 也写错了!!!
    //    Matrix<double, 2, 5> trks;
    //    trks << 841.20, 193.78, 917.58, 370.78, 0.00,
    //            5.75, 102.81, 80.40, 463.04, 0.00;
    //    double iou_threshold = 0.22136877277096445;
    //    Matrix<double, 2, 2> velocities;
    //    velocities << 0.00, -1.00, -0.45, 0.89;
    //    Matrix<double, 2, 5> k_observations;
    //    k_observations << 847.00, 193.00, 914.00, 371.00, 0.67, 0.00, 120.00, 81.00, 452.00, 0.81;
    //    double inertia = 0.39;
    //    // 调试用assocaite来debug
    //    auto result = associate(dets_first, trks, iou_threshold, velocities, k_observations, inertia);
    //    vector<Matrix<int, 1, Dynamic>> matched = std::get<0>(result);
    //    vector<int> unmatched_dets = std::get<1>(result);
    //    vector<int> unmatched_trks = std::get<2>(result);

    // 初始化一个OCsort对象
    OCSort A = OCSort(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, false);
    // 使用OCsort追踪，现在不停的给他传入我们的观测值(检测框)
    std::ostringstream filename;
    for (int i = 1; i < 526; ++i) {
        // 读取输入数据
        std::cout << "============== " << i << " =============" << std::endl;
        filename << "../BINARY_DATA/MOT17/" << i << ".csv";
        Eigen::Matrix<double, Eigen::Dynamic, 6> dets = read_csv_to_eigen(filename.str());
        filename.str("");
//        std::cout << "input:\n"
//                  << dets << std::endl;
        // 推理
        std::vector<Eigen::RowVectorXd> res = A.update(dets);
        // 打印输出
        std::cout << "predict:\n"
                  << res << std::endl;
        // 保存输出
        ofstream file;
        filename << "../OUTPUT_DATA/MOT17/" << i << ".txt";
        file.open(filename.str());
        filename.str("");
        IOFormat CSVFormat(StreamPrecision, DontAlignCols, ",", "\n");
        for (auto i: res) {
            file << i.format(CSVFormat) << endl;
        }
        file.close();
    }
//    std::cout << "==========Testing 118 Frame:===============\n"
//              << std::endl;
//    Eigen::Matrix<double, Eigen::Dynamic, 6> dets =
//            read_csv_to_eigen("../BINARY_DATA/118.csv");
//    A.update(dets);
}
