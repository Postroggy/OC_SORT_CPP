#pragma once
#include <opencv2/core/core.hpp>
struct Object
{
    // Generally it is the coordinates of the top left corner + width and height
    cv::Rect_<float> rect;
    int label;
    float prob;
};