#include "../include/OCsort.h"
#include "iomanip"
#include <utility>

namespace ocsort {
    /*重载 << for vector ，直接可以打印 vector*/
    template<typename Matrix>
    std::ostream &operator<<(std::ostream &os, const std::vector<Matrix> &v) {
        os << "{";
        for (auto it = v.begin(); it != v.end(); ++it) {
            os << "(" << *it << ")\n";
            if (it != v.end() - 1) os << ",";
        }
        os << "}\n";
        return os;
    }

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
        // 声明unordered_map，key为字符串，value为无参无返回值函数指针类型的function对象
        std::unordered_map<std::string, std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, const Eigen::MatrixXd &)>> ASSO_FUNCS{
                {"iou", iou_batch},
                {"giou", giou_batch}};
        ;
        // 确定之后需要使用的函数 , 虽然这个保存了函数指针，实际上我没有用到
        std::function<Eigen::MatrixXd(const Eigen::MatrixXd &, const Eigen::MatrixXd &)> asso_func = ASSO_FUNCS[asso_func_];
        // asso_func = std::move(asso_func_);// todo 难道这里我用函数指针实现？
        inertia = inertia_;
        use_byte = use_byte_;
        KalmanBoxTracker::count = 0;
    }
    // fixme: 这个是控制打印精度，发布时要删除的
    std::ostream &precision(std::ostream &os) {
        os << std::fixed << std::setprecision(2);
        return os;
    }
    std::vector<Eigen::RowVectorXd> OCSort::update(Eigen::MatrixXd dets) {
        /*
         * 输入矩阵dets: 形状 (n,5) 元素格式：[[x1,y1,x2,y2,confidence_score],...[...]]
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
        Eigen::Matrix<double, Eigen::Dynamic, 4> xyxys = dets.leftCols(4);
        Eigen::Matrix<double, 1, Eigen::Dynamic> confs = dets.col(4);// 这是 [1,n1]的行向量
        Eigen::Matrix<double, 1, Eigen::Dynamic> clss = dets.col(5); // 最后一列是 目标的类别
        Eigen::MatrixXd output_results = dets;
        auto inds_low = confs.array() > 0.1;
        auto inds_high = confs.array() < det_thresh;
        // 置信度在 0.1~det_thresh的需要二次匹配 inds_second => (1xN)
        auto inds_second = inds_low && inds_high;
        // 筛选一下，模拟 dets_second = output_results[inds_second];dets = output_results[remain_inds]
        Eigen::Matrix<double, Eigen::Dynamic, 6> dets_second;// 行不固定，列固定
        Eigen::Matrix<bool, 1, Eigen::Dynamic> remain_inds = (confs.array() > det_thresh);
        //因为python类型随便改而C++不行, note: 后续用 dets_first 代替dets传入 associate() 函数，切记！
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
        // (0,5)不会引起bug？ 是的，不会
        Eigen::MatrixXd trks = Eigen::MatrixXd::Zero(trackers.size(), 5);
        // 要删除的轨迹？但是后面不判断Nan，这个数组就没用了
        std::vector<int> to_del;
        std::vector<Eigen::RowVectorXd> ret;// 要返回的结果? 里面是 [1,7] 的行向量
        // 遍历 trks , 按行遍历
        for (int i = 0; i < trks.rows(); i++) {
            Eigen::RowVectorXd pos = trackers[i].predict();// predict 返回的结果应该是 (1,4) 的行向量
            trks.row(i) << pos(0), pos(1), pos(2), pos(3), 0;
            // note: 判断数据是不是 nan 的步骤我这里不写了，感觉基本不会有nan, 不判断Nan的话，前面这个变量  to_del 就用不到了
        }
        // 计算速度,shape：(n3,2)，用于 ORM，下面代码模拟python列表推导
        Eigen::MatrixXd velocities = Eigen::MatrixXd::Zero(trackers.size(), 2);
        Eigen::MatrixXd last_boxes = Eigen::MatrixXd::Zero(trackers.size(), 5);
        Eigen::MatrixXd k_observations = Eigen::MatrixXd::Zero(trackers.size(), 5);
        for (int i = 0; i < trackers.size(); i++) {
            velocities.row(i) = trackers[i].velocity;// 反正初始化为0了的，不用取判断is None了
            last_boxes.row(i) = trackers[i].last_observation;
            k_observations.row(i) = k_previous_obs(trackers[i].observations, trackers[i].age, delta_t);
        }
        /////////////////////////
        ///  Step1 First round of association
        ////////////////////////
        // 做iou关联  associate()
        std::vector<Eigen::Matrix<int, 1, 2>> matched;// 数组内 元素形状是(1,2)
        std::vector<int> unmatched_dets;
        std::vector<int> unmatched_trks;
        auto result = associate(dets_first, trks, iou_threshold, velocities, k_observations, inertia);
        matched = std::get<0>(result);
        unmatched_dets = std::get<1>(result);
        unmatched_trks = std::get<2>(result);
        // 把匹配上的 update
        for (auto m: matched) {
            // todo 用于 update 的向量是 (1,5) 的行向量， 但是VectorXd 是一个(5,1) 的列向量，这里会发生隐式转换
            Eigen::Matrix<double, 5, 1> tmp_bbox;
            tmp_bbox = dets_first.block<1, 5>(m(0), 0);
            trackers[m(1)].update(&(tmp_bbox), dets_first(m(0), 5));
        }

        ///////////////////////
        /// Step2 Second round of associaton by OCR to find lost tracks back
        //////////////////////
        // BYTE 的关联，5月4日 写的
        if (true == use_byte && dets_second.rows() > 0 && unmatched_trks.size() > 0) {
            Eigen::MatrixXd u_trks(unmatched_trks.size(), trks.cols());
            int index_for_u_trks = 0;
            for (auto i: unmatched_trks) {
                u_trks.row(index_for_u_trks++) = trks.row(i);
            }
            Eigen::MatrixXd iou_left = giou_batch(dets_second, u_trks);
            // 用map存的函数指针，替换一下
            // Eigen::MatrixXd iou_left = asso_func(dets_second, u_trks);
            if (iou_left.maxCoeff() > iou_threshold) {
                /**
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                 * */
                std::vector<std::vector<double>> iou_matrix(iou_left.rows(), std::vector<double>(iou_left.cols()));
                for (int i = 0; i < iou_left.rows(); i++) {
                    for (int j = 0; j < iou_left.cols(); j++) {
                        iou_matrix[i][j] = -iou_left(i, j);// note： 这里取反
                    }
                }
                // 进行线性分配
                std::vector<int> rowsol, colsol;
                double MIN_cost = execLapjv(iou_matrix, rowsol, colsol, true, 0.01, true);
                std::vector<std::vector<int>> matched_indices;
                // 为matched_indices赋值 version :0.1
                for (int i = 0; i < rowsol.size(); i++) {
                    if (rowsol.at(i) >= 0) {
                        matched_indices.push_back({colsol.at(rowsol.at(i)), rowsol.at(i)});
                    }
                }

                std::vector<int> to_remove_trk_indices;
                // 遍历线性分配的结果
                for (auto m: matched_indices) {
                    int det_ind = m[0];
                    int trk_ind = unmatched_trks[m[1]];
                    if (iou_left(m[0], m[1]) < iou_threshold) continue;

                    Eigen::Matrix<double, 5, 1> tmp_box;
                    tmp_box = dets_second.block<1, 5>(det_ind, 0);
                    trackers[trk_ind].update(&tmp_box, dets_second(det_ind, 5));
                    to_remove_trk_indices.push_back(trk_ind);
                }
                // 更新 unmatched_trks
                std::vector<int> tmp_res1(unmatched_trks.size());
                sort(unmatched_trks.begin(), unmatched_trks.end());              // 排序
                sort(to_remove_trk_indices.begin(), to_remove_trk_indices.end());// 排序
                auto end1 = set_difference(unmatched_trks.begin(), unmatched_trks.end(),
                                           to_remove_trk_indices.begin(), to_remove_trk_indices.end(),
                                           tmp_res1.begin());
                tmp_res1.resize(end1 - tmp_res1.begin());
                unmatched_trks = tmp_res1;// 更新
            }
        }


        if (unmatched_dets.size() > 0 && unmatched_trks.size() > 0) {
            // 模拟：left_dets = dets[unmatched_dets] 没匹配上轨迹的 检测
            Eigen::MatrixXd left_dets(unmatched_dets.size(), 6);
            // 哈哈，bug 修复了，left_dets.row(inx_for_dets++) = dets_first.row(i) 这两个的index不一样的
            int inx_for_dets = 0;
            for (auto i: unmatched_dets) {
                left_dets.row(inx_for_dets++) = dets_first.row(i);
            }
            // 模拟 left_trks = last_boxes[unmatched_trks] 最后一次匹配上的检测的轨迹
            Eigen::MatrixXd left_trks(unmatched_trks.size(), last_boxes.cols());
            int indx_for_trk = 0;
            for (auto i: unmatched_trks) {
                left_trks.row(indx_for_trk++) = last_boxes.row(i);
            }
            // 计算代价矩阵 todo: 这里暂时用 iou_batch 吧。后续再做映射了
            Eigen::MatrixXd iou_left = giou_batch(left_dets, left_trks);
            if (iou_left.maxCoeff() > iou_threshold) {
                /**
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                 * */
                // 找回丢失的track
                // todo：lapjv用的别人实现的库
                // 先把 iou_left 转成 二维 vector，转换的过程中元素取反
                std::vector<std::vector<double>> iou_matrix(iou_left.rows(), std::vector<double>(iou_left.cols()));
                for (int i = 0; i < iou_left.rows(); i++) {
                    for (int j = 0; j < iou_left.cols(); j++) {
                        iou_matrix[i][j] = -iou_left(i, j);// note： 这里取反
                    }
                }
                // 进行线性分配
                std::vector<int> rowsol, colsol;
                double MIN_cost = execLapjv(iou_matrix, rowsol, colsol, true, 0.01, true);
                //，下面生成 rematched_indices ,形状 nx2
                std::vector<std::vector<int>> rematched_indices;
                // 为rematched_indices赋值 version :0.1
                for (int i = 0; i < rowsol.size(); i++) {
                    if (rowsol.at(i) >= 0) {
                        rematched_indices.push_back({colsol.at(rowsol.at(i)), rowsol.at(i)});
                    }
                }
                // 假如重新线性分配了还没匹配上，那么这些都是需要删除的
                std::vector<int> to_remove_det_indices;
                std::vector<int> to_remove_trk_indices;
                // 遍历线性分配的结果
                for (auto i: rematched_indices) {
                    int det_ind = unmatched_dets[i.at(0)];
                    int trk_ind = unmatched_trks[i.at(1)];
                    if (iou_left(i.at(0), i.at(1)) < iou_threshold) {
                        continue;
                    }
                    ////////////////////////////////
                    ///  Step3  update status of second matched tracks
                    ///////////////////////////////
                    // fixme:这里更新，是因为又重新匹配上了
                    Eigen::Matrix<double, 5, 1> tmp_bbox;
                    tmp_bbox = dets_first.block<1, 5>(det_ind, 0);
                    trackers.at(trk_ind).update(&tmp_bbox, dets_first(det_ind, 5));
                    to_remove_det_indices.push_back(det_ind);
                    to_remove_trk_indices.push_back(trk_ind);
                }
                // 更新 unmatched_dets & trks ,模拟 setdiff1d 函数
                std::vector<int> tmp_res(unmatched_dets.size());
                // note: 因为 set_difference 要求必须是有序的再比较，将这些数据排序应该不会引起什么bug
                sort(unmatched_dets.begin(), unmatched_dets.end());              // 排序
                sort(to_remove_det_indices.begin(), to_remove_det_indices.end());// 排序
                auto end = set_difference(unmatched_dets.begin(), unmatched_dets.end(),
                                          to_remove_det_indices.begin(), to_remove_det_indices.end(),
                                          tmp_res.begin());
                tmp_res.resize(end - tmp_res.begin());
                unmatched_dets = tmp_res;// 更新
                std::vector<int> tmp_res1(unmatched_trks.size());
                sort(unmatched_trks.begin(), unmatched_trks.end());              // 排序
                sort(to_remove_trk_indices.begin(), to_remove_trk_indices.end());// 排序
                auto end1 = set_difference(unmatched_trks.begin(), unmatched_trks.end(),
                                           to_remove_trk_indices.begin(), to_remove_trk_indices.end(),
                                           tmp_res1.begin());
                tmp_res1.resize(end1 - tmp_res1.begin());
                unmatched_trks = tmp_res1;// 更新
            }
        }

        // fixme：论文中提到，如果现存的轨迹并没有观测值，那么也是要更新的
        for (auto m: unmatched_trks) {
            // python版本给 update 传入 z=None，但是在C++版本中，我们传入 nullptr 就行了
            trackers.at(m).update(nullptr, 0);
        }
        ///////////////////////////////
        /// Step4 Initialize new tracks and remove expired tracks
        ///////////////////////////////
        /*create and initialise new trackers for unmatched detections*/
        for (int i: unmatched_dets) {
            Eigen::RowVectorXd tmp_bbox = dets_first.block(i, 0, 1, 5);
            // todo 4.28 17:25 ,  dets(i, 5) 是目标的类别 cls
            int cls_ = int(dets(i, 5));
            KalmanBoxTracker trk = KalmanBoxTracker(tmp_bbox, cls_, delta_t);
            // 将新创建的追加到 trackers 末尾
            trackers.push_back(trk);
        }
        int tmp_i = trackers.size();// fixme: 不知道拿来干嘛的，好像是用来保存MOT格式的测试结果的
                                    // 逆序遍历 trackers 数组，生成需要返回的结果
        for (int i = trackers.size() - 1; i >= 0; i--) {
            // 下面是获取 预测值，有两种方式，差别其实不大
            Eigen::Matrix<double, 1, 4> d;
            int last_observation_sum = trackers.at(i).last_observation.sum();
            if (last_observation_sum < 0) {
                d = trackers.at(i).get_state();
            } else {
                /**
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                 * */
                d = trackers.at(i).last_observation.block(0, 0, 1, 4);
            }
            /**
                如果在一定阈值内（一般为1帧）未检测到物体，则将当前跟踪器标记为“未更新”
                判断条件 time_since_update < 1 意味着跟踪器在上一帧中有正确地匹配到目标并预测了当前帧中的位置，
                且在当前帧中的位置还没有超过一个阈值（1帧），说明跟踪器仍然有效，可以将其作为匹配结果加入到 ret 中
             * */
            if (trackers.at(i).time_since_update < 1 && ((trackers.at(i).hit_streak >= min_hits) | (frame_count <= min_hits))) {
                // +1 as MOT benchmark requires positive
                // d 是坐标 1x4 , trackers.at(i).id ，cls，conf 都是1x1标量
                // 将他们组合起来 (d,id,cls,conf) 形成 1x7 的行向量
                Eigen::RowVectorXd tracking_res(7);
                // note: ID 从 1 开始，符合MOT的格式
                tracking_res << d(0), d(1), d(2), d(3), trackers.at(i).id + 1, trackers.at(i).cls, trackers.at(i).conf;
                ret.push_back(tracking_res);// 主要是把数据组合起来，28号再写了
            }
            // remove dead tracklets
            if (trackers.at(i).time_since_update > max_age) {
                /*这里需要删除指定位置的元素*/
                trackers.erase(trackers.begin() + i);
            }
        }
        return ret;
    }
}// namespace ocsort