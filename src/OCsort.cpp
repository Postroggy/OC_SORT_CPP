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

        // 置信度在 0.1~det_thresh的需要二次匹配 inds_second => (1xN)
        auto inds_second = inds_low && inds_high;
        // 筛选一下，模拟 dets_second = output_results[inds_second];dets = output_results[remain_inds]
        Eigen::Matrix<double, Eigen::Dynamic, 6> dets_second;// 行不固定，列固定
        Eigen::Matrix<bool, 1, Eigen::Dynamic> remain_inds = (confs.array() > det_thresh);
        //因为python类型随便改而C++不行, note: 后续用 dets_second 代替dets传入 associate() 函数，切记！
        // todo: dets_first 出错了
        Eigen::Matrix<double, Eigen::Dynamic, 6> dets_first;
        for (int i = 0; i < output_results.rows(); i++) {
            if (true == inds_second(i)) {
                dets_second.conservativeResize(dets_second.rows() + 1, Eigen::NoChange);
                dets_second.row(dets_second.rows() - 1) = output_results.row(i);
            }
            if (true == remain_inds(i)) {
                dets_first.conservativeResize(dets_first.rows() + 1, Eigen::NoChange);
                dets_first.row(dets_first.rows() - 1) = output_results.row(i);
            }
        }

        /*get predicted locations from existing trackers.*/
        // (0,5)不会引起bug？ NO
        Eigen::MatrixXd trks = Eigen::MatrixXd::Zero(trackers.size(), 5);
        // 要删除的轨迹？但是后面不判断Nan，这个数组就没用了
        std::vector<int> to_del;
        std::vector<Eigen::VectorXd> ret;// 要返回的结果? 里面是 [1,7] 的向量
        // 遍历 trks , 按行遍历
        for (int i = 0; i < trks.rows(); i++) {
            Eigen::VectorXd pos = trackers[i].predict();// predict 返回的结果应该是(1,4) 的行向量
            trks.row(i) << pos(0), pos(1), pos(2), pos(3), 0;
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
        // 做iou关联  associate()
        std::vector<Eigen::Matrix<int, 1, Eigen::Dynamic>> matched;// 数组内 元素形状是(1,2)
        std::vector<int> unmatched_dets;
        std::vector<int> unmatched_trks;
        auto result = associate(dets_first, trks, iou_threshold, velocities, k_observations, inertia);
        matched = std::get<0>(result);
        unmatched_dets = std::get<1>(result);
        unmatched_trks = std::get<2>(result);
        // 把匹配上的 update
        for (auto m: matched) {
            // todo 用于 update 的向量是 (1,5) 的行向量， 但是VectorXd 是一个(5,1) 的列向量，update函数还得看下
            Eigen::VectorXd tmp_bbox = dets_first.block(m(0), 0, 1, 5);
            trackers[m(1)].update(&(tmp_bbox), dets_first(m(0), 5));
        }

        ///////////////////////
        /// Setp2 Second round of associaton by OCR to find lost tracks back
        //////////////////////
        // BYTE 的关联，暂时不用
        //        if(unmatched_dets.rows()> 0 && unmatched_trks.rows()>0){
        //
        //        }
        //        if(unmatched_dets.size()>0 && unmatched_trks.size()>0){
        //            Eigen::MatrixXd left_dets = dets_first;
        //        }
        if (unmatched_dets.size() > 0 && unmatched_trks.size() > 0) {
            // 模拟：left_dets = dets[unmatched_dets] 没匹配上轨迹的 检测
            Eigen::MatrixXd left_dets(unmatched_dets.size(), dets_first.cols());
            for (auto i: unmatched_dets) {
                left_dets.row(i) = dets_first.row(i);
            }
            // 模拟 left_trks = last_boxes[unmatched_trks] 最后一次匹配上的检测的轨迹
            Eigen::MatrixXd left_trks(unmatched_trks.size(), last_boxes.cols());
            for (auto i: unmatched_trks) {
                left_trks.row(i) = last_boxes.row(i);
            }
            // 计算代价矩阵 todo: 这里暂时用 iou_batch 吧。后续再做映射了
            Eigen::MatrixXd iou_left = iou_batch(left_dets, left_trks);
            // todo 今天先写到这了，27号，23:08
        }

        return xyxys;
    }
}// namespace ocsort