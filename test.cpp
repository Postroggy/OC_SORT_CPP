#include <Eigen/Dense>
#include <OCsort.h>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>


using namespace std;
using namespace Eigen;
using namespace ocsort;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

Eigen::Matrix<float, Eigen::Dynamic, 6> read_csv_to_eigen(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
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
    Eigen::Matrix<float, Eigen::Dynamic, 6> matrix(data.size(), data[0].size());
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}

int main(int argc, char *argv[]) {
    // Default test data path, can be overridden by command line argument
    // Note: Run this executable from the project root directory, or provide absolute path
    std::string csv_folder = "test_data/MOT17-01";
    int num_frames = 23;// MOT17-01 has 23 CSV files

    if (argc >= 2) {
        csv_folder = argv[1];
    }
    if (argc >= 3) {
        num_frames = std::stoi(argv[2]);
    }

    std::cout << "CSV folder: " << csv_folder << std::endl;
    std::cout << "Number of frames: " << num_frames << std::endl;
    std::cout << "(Run from project root directory or use absolute paths)" << std::endl;

    ocsort::OCSort tracker = ocsort::OCSort(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, true);

    double total_time = 0.0;
    int processed_frames = 0;

    for (int i = 1; i <= num_frames; i++) {
        std::ostringstream filename;
        filename << csv_folder << "/" << i << ".csv";

        // Check if file exists
        std::ifstream test_file(filename.str());
        if (!test_file.good()) {
            std::cout << "File not found: " << filename.str() << ", stopping." << std::endl;
            break;
        }
        test_file.close();

        Eigen::Matrix<float, Eigen::Dynamic, 6> dets = read_csv_to_eigen(filename.str());

        auto t_start = high_resolution_clock::now();
        std::vector<Eigen::RowVectorXf> results = tracker.update(dets);
        auto t_end = high_resolution_clock::now();

        duration<double, std::milli> ms_double = t_end - t_start;
        total_time += ms_double.count();
        processed_frames++;

        // Print tracking results for current frame
        std::cout << "Frame " << i << ": " << results.size() << " tracked objects, ";
        std::cout << "time: " << ms_double.count() << "ms" << std::endl;
    }

    if (processed_frames > 0) {
        double avg_cost = total_time / processed_frames;
        int fps = static_cast<int>(1000.0 / avg_cost);
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Processed frames: " << processed_frames << std::endl;
        std::cout << "Total time: " << total_time << "ms" << std::endl;
        std::cout << "Average time per frame: " << avg_cost << "ms" << std::endl;
        std::cout << "Average FPS: " << fps << std::endl;
    }

    return 0;
}
