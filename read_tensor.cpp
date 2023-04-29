/* 该文件的目的是测试C++的Eigen读取torch保存的pkl数据*/
#include "iostream"
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;
using namespace Eigen;
/**
 * 读取CSV文件，并且返回一个Matrix
 * @param filename 
 * @return 
 */
Eigen::Matrix<double,Eigen::Dynamic,6> read_csv_to_eigen(const std::string &filename) {
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

int main(int argc, char *argv[]) {
    // 读取CSV文件，转成Matrix
    Eigen::Matrix<double,Eigen::Dynamic,6> matrix = read_csv_to_eigen("../BINARY_DATA/1.csv");

    return 0;
}