#include <Eigen/Dense>
#include <OCsort.h>
#include <chrono>// 用于计算耗时
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>


using namespace std;
using namespace Eigen;
using namespace ocsort;
/* 用于计时 */
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

Eigen::Matrix<float, Eigen::Dynamic, 6> read_csv_to_eigen(const std::string &filename) {
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
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}
/**
 * @brief 将Vector转为Matrix
 * 
 * @param data 
 * @return Eigen::Matrix<float, Eigen::Dynamic, 6> 
 */
Eigen::Matrix<float, Eigen::Dynamic, 6> Vector2Matrix(std::vector<std::vector<float>> data) {
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}
// 重载 << 输出 vector
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

// 将读取的一行数据转为vector，并且从 tlwh=>xyxys
std::vector<float> String2Vector(std::string line, bool Format = true) {
    std::vector<float> x;
    std::stringstream ss(line);
    std::string item;
    std::vector<float> data;
    while (std::getline(ss, item, ',')) {
        data.push_back(std::stod(item));
    }
    // 取完这一行的data，我们需要转化一下格式
    float x1 = data[2];
    float y1 = data[3];
    float x2 = data[2] + data[4];
    float y2 = data[3] + data[5];
    // 我们需要的数据格式：xyxys,score,0
    x.insert(x.end(), {x1, y1, x2, y2, data[6], 0});
    if (Format)
        return x;
    else
        return data;
}

int main(int argc, char *argv[]) {
    // 创建文件句柄
    std::ostringstream filename;
    // 读取数据
    std::string filePath;
    std::cout<<"Plz enter the MOT data path: ";
    std::cin>>filePath;
    filename << filePath;
    std::ifstream fileA(filename.str());
    if (fileA.is_open()) {
        std::cout << "File is Opened, Path is:" << filename.str() << "\n";
        // 读取文本
        std::string line;
        int flag = 1;
        // 究极大矩阵
        std::vector<Eigen::MatrixXf> ALL_INPUT;
        // 每一帧的矩阵
        Eigen::MatrixXf Frame_INPUT;
        std::vector<std::vector<float>> Frame;
        // 存上一个
        std::vector<float> Frame_Previous;
        std::vector<float> tmp_fmt;
        std::vector<float> tmp;
        // for (int i = 0; i < 104; i++) {
        while (true) {
            std::getline(fileA, line);
            // 因为最后一帧的目标肯定不止一个，所以就只需要在这部分判断是否到文件结尾了
            if (fileA.eof()) {
                // 读到结尾，把之前所有的都转化为矩阵
                std::cout << "Read the END OF FILES" << std::endl;
                if (Frame_Previous.size() != 0)
                    Frame.push_back(Frame_Previous);
                ALL_INPUT.push_back(Vector2Matrix(Frame));
                break;
            }
            tmp_fmt = String2Vector(line);
            tmp = String2Vector(line, false);// 不转化格式的string转vector
            // std::cout << tmp[0] << std::endl;
            if ((int) tmp[0] != flag) {
                // 说明已经到了下一帧了
                flag = (int) tmp[0];
                // 保存这一行的数据
                Frame_Previous = tmp_fmt;
                // 将上一帧的数据全部保存为Matrix
                Frame_INPUT = Vector2Matrix(Frame);
                ALL_INPUT.push_back(Frame_INPUT);
                // std::cout << Frame << std::endl;
                // 清除上一帧的数据
                Frame.clear();
            } else {                             // 还能继续把这一行的数据push进去
                if (Frame_Previous.size() != 0) {// 如果上一行有记录的话，那么把他push_back
                    Frame.push_back(Frame_Previous);
                    Frame_Previous.clear();
                }
                Frame.push_back(tmp_fmt);
            }
        }
        // 至此，所有的Frame数据，都被存在了ALL_INPUT里面
        std::cout << "Size of data: " << ALL_INPUT.size() << std::endl;
        ocsort::OCSort A = ocsort::OCSort(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, true);
        float OverAll_Time = 0.0;
        // 遍历所有的输入，送到OCSORT里面
        for (auto dets: ALL_INPUT) {
            auto T_start = high_resolution_clock::now();
            std::vector<Eigen::RowVectorXf> res = A.update(dets);
            auto T_end = high_resolution_clock::now();
            duration<float, std::milli> ms_float = T_end - T_start;
            OverAll_Time += ms_float.count();
        }
        // 计算平均帧率。
        float avg_cost = OverAll_Time / ALL_INPUT.size();
        int FPS = int(1000 / avg_cost);
        std::cout << "Average Time Cost per Frame: " << avg_cost << " Avg FPS: " << FPS << std::endl;
    } else
        std::cout << "open Failed" << std::endl;

    return 0;
}
