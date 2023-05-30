#include <Eigen/Dense>
#include <OCsort.h>
#include <chrono> // 用于计算耗时
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vector>


/* 用于计时 */
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

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
std::ostream &operator<<(std::ostream &os, const std::vector<AnyCls> &v) {
    os << "{";
    for (auto it = v.begin(); it != v.end(); ++it) {
        os << "(" << *it << ")";
        if (it != v.end() - 1) os << ", ";
    }
    os << "}";
    return os;
}


int main(int argc, char *argv[]) {
    ocsort::OCSort A = ocsort::OCSort(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, true);
    std::ostringstream filename;
    // 打开视频
    // cv::VideoCapture cap("C://SOTA//DATASET//MOT17//train//MOT17-02-FRCNN//img1//output.mp4");
    // if (!cap.isOpened()) {
    //     std::cout << "Error opening video file" << std::endl;
    //     return -1;
    // }
    // 临时帧
    cv::Mat frame;
    // 记录总耗时
    double OverAll_Time = 0;
    for (int i = 1; i < 526; i++) {
        // std::cout << "============== " << i << " =============" << std::endl;
        filename.str("");
        filename << "C:/SOTA/figure_ocsort/MOT_xyxys/private/SeedDet/MOT17-02/" << i << ".csv";
        // filename << "../test_data/MOT17-01/" << i << ".csv";
        Eigen::Matrix<double, Eigen::Dynamic, 6> dets = read_csv_to_eigen(filename.str());
        auto T_start = high_resolution_clock::now();
        std::vector<Eigen::RowVectorXd> res = A.update(dets);
        auto T_end = high_resolution_clock::now();
        auto ms_int = duration_cast<milliseconds>(T_end - T_start);
        duration<double, std::milli> ms_double = T_end - T_start;
        // 打印计算耗时
        // std::cout << "computation cost: " << ms_int.count() << " ms" << std::endl;
        // 累加时间
        OverAll_Time += ms_double.count();
        // 将追踪结果显示在桌面上
        // cap.read(frame);
        // // 绘制追踪结果，输出格式：左上、右下、ID、class、conf
        // for (auto j: res) {
        //     int ID = int(j[4]);
        //     int Class = int(j[5]);
        //     float conf = j[6];
        //     cv::putText(frame, cv::format("ID:%d", ID), cv::Point(j[0], j[1] - 5), 0, 0.5, cv::Scalar(229, 115, 115), 2, cv::LINE_AA);
        //     cv::rectangle(frame, cv::Rect(j[0], j[1], j[2] - j[0] + 1, j[3] - j[1] + 1), cv::Scalar(3, 155, 229), 2);
        // }
        // cv::imshow("Video", frame);// 显示帧
        // if (cv::waitKey(1) == 27) {// 按下ESC推出
        //     std::cout << "Program Terminate" << std::endl;
        //     cap.release();
        //     cv::destroyAllWindows();
        //     return 0;
        // }
    }
    // 计算平均帧率。
    double avg_cost = OverAll_Time/526;
    int FPS = int(1000/avg_cost);
    std::cout<<"Average Time Cost: "<<avg_cost<<" Avg FPS: "<<FPS<<std::endl;
    return 0;
}
