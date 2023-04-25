#ifndef OC_SORT_CPP_UTILITIES_H
#define OC_SORT_CPP_UTILITIES_H
#pragma once
#include "Eigen/Dense"
namespace ocsort {
    /**
 * Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
[x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
the aspect ratio
 * @param bbox
 * @return z
 */
    Eigen::VectorXd convert_bbox_to_z(Eigen::VectorXd bbox) {
        double w = bbox[2] - bbox[0];
        double h = bbox[3] - bbox[1];
        double x = bbox[0] + h / 2.0;
        double y = bbox[1] + h / 2.0;
        double s = w * h;
        double r = w / (h + 1e-6);
        Eigen::MatrixXd z(4, 1);
        z << x, y, s, r;
        return z;
    }
}
#endif//OC_SORT_CPP_UTILITIES_H
