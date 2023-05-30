#include <chrono>// 用于计算耗时
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


int main(int argc, char *argv[]) {
    std::ifstream fileA("C:/SOTA/figure_ocsort/SeedDet/MOT17-02.txt");
    std::string line;
    int flag=1;
    for (int i = 0; i < 100; i++) {
        std::streampos old_pos = fileA.tellg();
        std::getline(fileA, line);
        std::cout << line << std::endl;
        
        if (std::stoi(&line[0])!=flag) {
            std::cout<<"Flag:"<<flag<<" Line:"<<std::stoi(&line[0])<<std::endl;
            fileA.clear();
            std::cout<<"Move:"<<static_cast<std::streamoff>(old_pos) - i<<std::endl;
            fileA.seekg(static_cast<std::streamoff>(old_pos) - i);
            flag=std::stoi(&line[0]);
        }
    }
    return 0;
}