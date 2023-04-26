#include "../include/OCsort.h"
#include "iomanip"
#include <utility>
namespace ocsort {
    OCSort::OCSort(float det_thresh_, int max_age_, int min_hits_, float iou_threshold_, int delta_t_, std::string asso_func_, float inertia_, bool use_byte_) {
        /*Sets key parameters for SORT*/
        max_age = max_age_;
        min_hits = min_hits_;
        iou_threshold = iou_threshold_;
        trackers.clear();
        frame_count = 0;
        // 下面是 ocsort 新增的
        det_thresh = det_thresh_;
        delta_t = delta_t_;
        asso_func = std::move(asso_func_);// todo 难道这里我用函数指针实现？
        inertia = inertia_;
        use_byte = use_byte_;
        KalmanBoxTracker::count = 0;
    }
    Eigen::MatrixXd OCSort::update(Eigen::MatrixXd dets) {
        /*
         * Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        fixme：原版的函数原型：def update(self, output_results, img_info, img_size)
            这部分代码差异较大，具体请参考：https://www.diffchecker.com/fqTqcBSR/
         */
        frame_count += 1;
        /*下面这些都是临时变量*/
        Eigen::MatrixXd xyxys = dets.leftCols(4);
        Eigen::Matrix<double, 1, Eigen::Dynamic> confs = dets.col(4);// 这是 [1,n1]的行向量
        Eigen::Matrix<double, 1, Eigen::Dynamic> clss = dets.col(5); // 最后一列是 目标的类别
        Eigen::MatrixXd output_results = dets;
        // todo: 这里我没看明白,auto类型也迷迷糊糊的
        auto inds_low = confs.array() > 0.1;
        auto inds_high = confs.array() < det_thresh;
        // 置信度在 0.1~det_thresh的需要二次匹配
        auto inds_second = inds_low && inds_high;
        // 筛选一下，模拟 dets_second = output_results[inds_second];dets = output_results[remain_inds]
        Eigen::Matrix<double, Eigen::Dynamic, 6> dets_second;// 行不固定，列固定
        Eigen::Matrix<double, Eigen::Dynamic, 6> dets_first; //因为python类型随便改而C++不行，
        int index1 = 0, index2 = 0;
        for (int i = 0; i < output_results.rows(); i++) {
            if (true == inds_second(0, i)) {
                dets_second.resize(index1 + 1, 6);
                dets_second.row(index1) = output_results.row(i);
                index1++;
            } else {
                dets_first.resize(index2 + 1, 6);
                dets_first.row(index2) = output_results.row(i);
                index2++;
            }
        }
        /*get predicted locations from existing trackers.*/
        Eigen::MatrixXd trks = Eigen::MatrixXd::Zero(trackers.size(), 5);// (0,5)不会引起bug
        // 要删除的轨迹？但是后面不判断Nan，这个数组就没用了
        std::vector<int> to_del;
        std::vector<Eigen::VectorXd> ret;// 要返回的结果? 里面是 [1,7] 的向量
        // 遍历 trks , 按行遍历
        for (int i = 0; i < trks.rows(); i++) {
            Eigen::VectorXd pos = trackers[i].predict();// predict 返回的结果应该是(1,4) 的行向量
            trks.row(i) << pos(0), pos(1), pos(2), pos(3), 0;
            std::cout << trks.row(i) << std::endl;
            // 判断数据是不是 nan 的步骤我这里不写了，感觉基本不会有nan,
        }
        // 计算速度,shape：(n3,2)，用于ORM，下面代码模拟列表推导
        Eigen::MatrixXd velocities = Eigen::MatrixXd::Zero(trackers.size(), 2);
        Eigen::MatrixXd last_boxes = Eigen::MatrixXd::Zero(trackers.size(), 5);
        Eigen::MatrixXd k_observations = Eigen::MatrixXd::Zero(trackers.size(), 5);
        for (int i = 0; i < trackers.size(); i++) {
            velocities.row(i) = trackers[i].velocity;// 反正初始化为0了的，不用取判断is None了
            last_boxes.row(i) = trackers[i].last_observation;
            k_observations.row(i) = k_previous_obs(trackers[i].observations, trackers[i].age, delta_t);
        }
        /////////////////////////
        ///  Setp1 First round of association
        ////////////////////////
        Eigen::MatrixXd matched,unmatched_dets,unmatched_trks;
        // todo 做iou关联  associate()
        // 把匹配上的update
//        for(int i =0;i <matched.rows();i++){
//            trackers[matched(i)].update(,dets(1,2));
//        }
        ///////////////////////
        /// Setp2 Second round of associaton by OCR to find lost tracks back
        //////////////////////
        // BYTE 的关联，暂时不用
//        if(unmatched_dets.rows()> 0 && unmatched_trks.rows()>0){
//
//        }

        return xyxys;
    }
}// namespace ocsort