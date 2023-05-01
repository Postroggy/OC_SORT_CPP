#include <Eigen/Dense>
#include <OCsort.h>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
using namespace Eigen;
using namespace ocsort;
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
    std::ifstream file;
    file.open("../MOT17/test/MOT17-01-DPM/det/det.txt", ios::in);
    // 通过for循环不断的读取一帧中的信息
    if (!file.is_open()) {
        std::cerr << "File can not open !" << std::endl;
        exit(1);
    }
    // 解析csv内容
    std::string line;
    std::vector<double> row;
    std::vector<double> old_row;
    std::vector<std::vector<double>> data;// 存储每一帧的数据
    std::string field;
    // 解析文本，提取每一帧的信息
    for (int i = 1; i < 20; i++) {
        while (true) {
            // 读取一行之前先存他原来的position
            std::getline(file, line);
            // 如果上一次有读取到这一次的数据，则加进来
            if (old_row.size() > 0) {
                data.push_back(old_row);
                old_row.clear();
            }
            if (line.size() == 0)
                break;
            // 用于流式读取文本
            std::istringstream iss(line);
            line.clear();
            while (std::getline(iss, field, ',')) {
                row.push_back(std::stod(field));// 字符串转double浮点型
            }
            // 判断是否是这一帧的数据
            if (row.size() > 0 && int(row.at(0)) == i) {
                data.push_back(row);
                row.clear();
            }
            // 数据是按照1.2.3.有序排列的，发现对不上时，说明这是读取到下一帧的数据了
            else {
                // 将 file 指针 rollback
                old_row.swap(row);
                break;
            }
        }
        // 转换为Eigen::Matrix
        /* 数据格式：
              0   1     2      3       4        5       6   7 8 9
            frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z
            我们只需要： bb_left,bb_top,bb_width,bb_height,conf 就OK，class我们用0填充
         */
        Eigen::MatrixXd matrix(data.size(), 6);
        for (int i = 0; i < data.size(); ++i) {
            // MOT给出的格式(左上x,左上y,w,h)和我们需要的 (左上,右下) 格式有区别，转换一下
            // 右下x = 左上x+w ; 右下y = 左上y-h;
            matrix.row(i) << data[i][2], data[i][3], data[i][2] + data[i][4], data[i][3] - data[i][5], data[i][6], 0;
        }
        // 清空data，准备下一帧的数据读取
        data.clear();
        // 做预测：
        std::vector<Eigen::RowVectorXd> res = A.update(matrix);
        std::cout << "=========== REsult: " << i << "===============" << std::endl;
        std::cout << res << std::endl;
    }
    file.close();
    return 0;
}