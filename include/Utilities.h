#ifndef OC_SORT_CPP_UTILITIES_H
#define OC_SORT_CPP_UTILITIES_H
#include "Eigen/Dense"
namespace ocsort {
/**
 * Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
[x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
the aspect ratio
 * @param bbox
 * @return z
 */
    Eigen::VectorXd convert_bbox_to_z(Eigen::VectorXd bbox);
    Eigen::VectorXd speed_direction(Eigen::VectorXd bbox1, Eigen::VectorXd bbox2);
    Eigen::VectorXd convert_x_to_bbox(Eigen::VectorXd x) ;
    Eigen::VectorXd k_previous_obs(std::unordered_map<int, Eigen::VectorXd> observations_, int cur_age, int k);
}// namespace ocsort
#endif//OC_SORT_CPP_UTILITIES_H
