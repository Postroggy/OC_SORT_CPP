#include <Eigen/Dense>
#include <OCsort.h>
#include <cassert>
#include <chrono>// for timing
#include <fstream>
#include <iostream>
#include <vector>



using namespace std;
using namespace Eigen;
using namespace ocsort;
/* for timing */
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

Eigen::Matrix<float, Eigen::Dynamic, 6> read_csv_to_eigen(const std::string &filename) {
    // Read CSV file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
        exit(1);
    }
    // Parse CSV format
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
    // Convert to Eigen::Matrix
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}
/**
 * @brief Convert Vector to Matrix
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
// Overload << operator for vector output
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

// Convert a line of data to vector, and convert from tlwh => xyxys format
std::vector<float> String2Vector(std::string line, bool Format = true) {
    std::vector<float> x;
    std::stringstream ss(line);
    std::string item;
    std::vector<float> data;
    while (std::getline(ss, item, ',')) {
        data.push_back(std::stod(item));
    }
    // After parsing this line, we need to convert the format
    float x1 = data[2];
    float y1 = data[3];
    float x2 = data[2] + data[4];
    float y2 = data[3] + data[5];
    // Required data format: xyxys, score, 0
    x.insert(x.end(), {x1, y1, x2, y2, data[6], 0});
    if (Format)
        return x;
    else
        return data;
}

int main(int argc, char *argv[]) {
    // Create file handle
    std::ostringstream filename;
    // Use default test data path
    std::string filePath = "test_data/MOT17-02.txt";
    filename << filePath;
    std::ifstream fileA(filename.str());
    if (fileA.is_open()) {
        std::cout << "File is Opened, Path is:" << filename.str() << "\n";
        // Read text
        std::string line;
        int flag = 1;
        // Container for all frames
        std::vector<Eigen::MatrixXf> ALL_INPUT;
        // Matrix for each frame
        Eigen::MatrixXf Frame_INPUT;
        std::vector<std::vector<float>> Frame;
        // Store previous frame data
        std::vector<float> Frame_Previous;
        std::vector<float> tmp_fmt;
        std::vector<float> tmp;
        // for (int i = 0; i < 104; i++) {
        while (true) {
            std::getline(fileA, line);
            // Check for EOF here since the last frame has multiple detections
            if (fileA.eof()) {
                // Reached end, convert all remaining data to matrix
                std::cout << "Read the END OF FILES" << std::endl;
                if (Frame_Previous.size() != 0)
                    Frame.push_back(Frame_Previous);
                ALL_INPUT.push_back(Vector2Matrix(Frame));
                break;
            }
            tmp_fmt = String2Vector(line);
            tmp = String2Vector(line, false);// String to vector without format conversion
            // std::cout << tmp[0] << std::endl;
            if ((int) tmp[0] != flag) {
                // This means we've moved to the next frame
                flag = (int) tmp[0];
                // Save current line data
                Frame_Previous = tmp_fmt;
                // Save previous frame data as Matrix
                Frame_INPUT = Vector2Matrix(Frame);
                ALL_INPUT.push_back(Frame_INPUT);
                // std::cout << Frame << std::endl;
                // Clear previous frame data
                Frame.clear();
            } else {                             // Continue pushing data for current frame
                if (Frame_Previous.size() != 0) {// If there's previous record, push it back
                    Frame.push_back(Frame_Previous);
                    Frame_Previous.clear();
                }
                Frame.push_back(tmp_fmt);
            }
        }
        // Now all frame data is stored in ALL_INPUT
        std::cout << "Size of data: " << ALL_INPUT.size() << std::endl;
        ocsort::OCSort A = ocsort::OCSort(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, true);
        float OverAll_Time = 0.0;
        // Iterate through all inputs and feed to OCSORT
        for (auto dets: ALL_INPUT) {
            auto T_start = high_resolution_clock::now();
            std::vector<Eigen::RowVectorXf> res = A.update(dets);
            auto T_end = high_resolution_clock::now();
            duration<float, std::milli> ms_float = T_end - T_start;
            OverAll_Time += ms_float.count();
        }
        // Calculate average FPS
        float avg_cost = OverAll_Time / ALL_INPUT.size();
        int FPS = int(1000 / avg_cost);
        std::cout << "Average Time Cost per Frame: " << avg_cost << "ms" << " Avg FPS: " << FPS << std::endl;
    } else
        std::cout << "open Failed" << std::endl;

    return 0;
}
