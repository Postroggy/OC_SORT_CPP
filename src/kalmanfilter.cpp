#include "../include/kalmanfilter.h"
#include <iostream>
namespace ocsort {
    KalmanFilterNew::KalmanFilterNew(){};
    KalmanFilterNew::KalmanFilterNew(int dim_x_, int dim_z_) {
        dim_x = dim_x_;
        dim_z = dim_z_;
        x = Eigen::VectorXd::Zero(dim_x_, 1);
        // fixme 暂时数据类型是 double 的 P Q B F都初始化为单位矩阵
        P = Eigen::MatrixXd::Identity(dim_x_, dim_x_);
        Q = Eigen::MatrixXd::Identity(dim_x_, dim_x_);
        B = Eigen::MatrixXd::Identity(dim_x_, dim_x_);// 一般用不到
        F = Eigen::MatrixXd::Identity(dim_x_, dim_x_);
        H = Eigen::MatrixXd::Zero(dim_z_, dim_x_);
        R = Eigen::MatrixXd::Identity(dim_z_, dim_z_);
        M = Eigen::MatrixXd::Zero(dim_x_, dim_z_);
        z = Eigen::VectorXd::Zero(dim_z_, 1);
        /*
            gain and residual are computed during the innovation step. We
            save them so that in case you want to inspect them for various
            purposes
             * */
        K = Eigen::MatrixXd::Zero(dim_x_, dim_z_);
        y = Eigen::VectorXd::Zero(dim_x_, 1);
        S = Eigen::MatrixXd::Zero(dim_z_, dim_z_);
        // SI是S的转置，为什么不直接 S.T 呢？
        SI = Eigen::MatrixXd::Zero(dim_z_, dim_z_);

        // 下面是保存predict后的变量
        x_prior = x;
        P_prior = P;
        // 下面是保存update后的变量
        x_post = x;
        P_post = P;
    };
    void KalmanFilterNew::predict() {
        /**Predict next state (prior) using the Kalman filter state propagation
    equations. 这个函数需要传参，但是在ocsort中，没有传参，所以这是无需传参的版本，后续可以重载
    Parameters
    ----------
    u : np.array, default 0
        Optional control vector.
    B : np.array(dim_x, dim_u), or None
        Optional control transition matrix; a value of None
        will cause the filter to use `self.B`.
    F : np.array(dim_x, dim_x), or None
        Optional state transition matrix; a value of None
        will cause the filter to use `self.F`.
    Q : np.array(dim_x, dim_x), scalar, or None
        Optional process noise matrix; a value of None will cause the
        filter to use `self.Q`.
        */
        // x = Fx+Bu , 但是这里我不考虑 u 和 B
        x = F * x;
        // P = FPF' + Q , 其中F'是F的转置
        P = _alpha_sq * ((F * P), F.transpose()) + Q;
        // 保存之前的值
        x_prior = x;
        P_prior = P;
    }
    void KalmanFilterNew::update(Eigen::VectorXd *z_) {
        // 函数原型：update(self, z, R=None, H=None)，但是R H这里用不到
        // 传递的参数是指针类型的才好用空指针代替python中的 None
        /*Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
            If you pass in a value of H, z must be a column vector the
            of the correct size.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
         * */
        // NOTE:说明一下，z_ 的形状是 [dim_z,1] 的即[4,1]
        // 将新来的观测值加入到 history_obs 中,数组中存的是 Eigen::VectorXd的指针
        history_obs.push_back(z_);
        if (z_ == nullptr) {// 说明轨迹追踪不到目标了
            if (true == observed) freeze();
            observed = false;// ocsort 新增
            // 原py代码中：z = np.array([ [None]*dim_z ]).T
            z = Eigen::VectorXd::Zero(dim_z, 1);
            x_post = x;
            P_post = P;
            y = Eigen::VectorXd::Zero(dim_z, 1);
            return;
        }
        // 如果又接受到观测值了，那么解冻
        /*
            Get observation, use online smoothing to re-update parameters => OOS
            现在这一步叫 ORU
            observed为False=>接收到观测值了。则解冻。
         */
        if (false == observed) unfreeze();
        observed = true;
        // 下面是正经的更新算法
        // y = z - Hx , 残差 between measurement观测 and prediction预测
        y = *z_ - H * x;
        // 下面这个表达式PH' 因为在计算S和K都有用到，所以先保存，减少计算量
        auto PHT = P * H.transpose();
        // S = HPH' + R, 其中 H'是H的转置,S是观测残差的协方差
        S = H * PHT + R;
        // S的逆矩阵，后续计算要用到
        SI = S.inverse();
        // K = PH'SI ，计算最优kalman增益
        K = PHT * SI;
        // note:更新x
        x = x + K * y;
        // note:更新P
        /*This is more numerically stable and works for non-optimal K vs the
         * equation P = (I-KH)P usually seen in the literature.*/
        auto I_KH = I - K * H;
        P = ((I_KH * P) * I_KH.transpose()) + ((K * R) * K.transpose());
        // save the measurement and posterior state
        z = *z_;
        x_post = x;
        P_post = P;
    }
    void KalmanFilterNew::freeze() {
        /*将这个对象中的所有变量都保存起来
         * save all the variable in current object at the time
         * */
        // note:最重要的一点，先将标志置true
        attr_saved.IsInitialized = true;
        ///////////下面开始存档///////////
        attr_saved.x = x;
        attr_saved.P = P;
        attr_saved.Q = Q;
        attr_saved.B = B;
        attr_saved.F = F;
        attr_saved.H = H;
        attr_saved.R = R;
        attr_saved._alpha_sq = _alpha_sq;
        attr_saved.M = M;
        attr_saved.z = z;
        attr_saved.K = K;
        attr_saved.y = y;
        attr_saved.S = S;
        attr_saved.SI = SI;
        attr_saved.x_prior = x_prior;
        attr_saved.P_prior = P_prior;
        attr_saved.x_post = x_post;
        attr_saved.P_post = P_post;
        attr_saved.history_obs = history_obs;
    }
    void KalmanFilterNew::unfreeze() {
        /* 将freeze保存的变量全部加载进来 */
        // todo：后续记得把 attr_saved.size > 0条件 加进来
        if (true == attr_saved.IsInitialized) {
            std::cout << "正在恢复轨迹" << std::endl;
            new_history = history_obs;
            /*开始数据恢复*/
            x = attr_saved.x;
            P = attr_saved.P;
            Q = attr_saved.Q;
            B = attr_saved.B;
            F = attr_saved.F;
            H = attr_saved.H;
            R = attr_saved.R;
            _alpha_sq = attr_saved._alpha_sq;
            M = attr_saved.M;
            z = attr_saved.z;
            K = attr_saved.K;
            y = attr_saved.y;
            S = attr_saved.S;
            SI = attr_saved.SI;
            x_prior = attr_saved.x_prior;
            P_prior = attr_saved.P_prior;
            x_post = attr_saved.x_post;
            /*数据恢复完成*/
            // 除了最后一次的obs，将之前的赋给history_obs，相当于在 history_obs 中除去最后一个观测值
            history_obs.erase(history_obs.end() - 1);
            // 因为 new_history 中有一部分可能是 nullptr，所以需要过滤一下,
            // 这里我们只需要取得最后两个有效的观测值
            // box1 = new_history[index1] # 从历史记录中取倒数第二个真实的观测值
            Eigen::VectorXd box1;           // 倒数第二个
            Eigen::VectorXd box2;           // 倒数第一个观测值
            int lastNotNullIndex = -1;      // 倒数第一个的index
            int secondLastNotNullIndex = -1;// 倒数第二个的index
            for (int i = new_history.size() - 1; i >= 0; i--) {
                if (new_history[i] != nullptr) {
                    if (lastNotNullIndex == -1) {
                        // 当前元素为最后一个非nullptr的元素
                        lastNotNullIndex = i;
                        box2 = *(new_history.at(lastNotNullIndex));
                    } else if (secondLastNotNullIndex == -1) {
                        // 当前元素为倒数第二个非nullptr的元素
                        secondLastNotNullIndex = i;
                        box1 = *(new_history.at(secondLastNotNullIndex));
                        break;
                    }
                }
            }
            // 计算 \Delta{t}
            double time_gap = lastNotNullIndex - secondLastNotNullIndex;
            // 提取数据，计算宽高
            double x1 = box1[0];
            double x2 = box2[0];
            double y1 = box1[1];
            double y2 = box2[1];
            double w1 = std::sqrt(box1[2] * box1[3]);
            double h1 = std::sqrt(box1[2] / box1[3]);
            double w2 = std::sqrt(box1[2] * box1[3]);
            double h2 = std::sqrt(box1[2] / box1[3]);
            // 计算对时间的导数
            double dx = (x2 - x1) / time_gap;// X 轴方向速度
            double dy = (y1 - y2) / time_gap;// Y 轴方向速度
            double dw = (w2 - w1) / time_gap;// w 对时间的变化率
            double dh = (h2 - h1) / time_gap;// h 对时间的变化率

            for (int i = 0; i < time_gap; i++) {
                /*
                    The default virtual trajectory generation is by linear
                    motion (constant speed hypothesis), you could modify this
                    part to implement your own.
                 */
                double x = x1 + (i + 1) * dx;
                double y = y1 + (i + 1) * dy;
                double w = w1 + (i + 1) * dw;
                double h = h1 + (i + 1) * dh;
                double s = w * h;
                double r = w / (h * 1.0);
                Eigen::VectorXd new_box(4, 1);
                new_box << x, y, s, r;
                /*
                    I still use predict-update loop here to refresh the parameters,
                    but this can be faster by directly modifying the internal parameters
                    as suggested in the paper. I keep this naive but slow way for
                    easy read and understanding
                    NOTE: 这里我就直接更新参数吧。
                 */
                // 下面是正经的更新算法
                // y = z - Hx , 残差 between measurement 观测 and prediction 预测
                this->y = new_box - this->H * this->x;
                // 下面这个表达式PH' 因为在计算S和K都有用到，所以先保存，减少计算量
                auto PHT = this->P * this->H.transpose();
                // S = HPH' + R, 其中 H'是H的转置,S是观测残差的协方差
                this->S = this->H * PHT + this->R;
                // S的逆矩阵，后续计算要用到
                this->SI = (this->S).inverse();
                // K = PH'SI ，计算最优kalman增益
                this->K = PHT * this->SI;
                // note:更新x
                this->x = this->x + this->K * this->y;
                // note:更新P
                /*This is more numerically stable and works for non-optimal K vs the
                 * equation P = (I-KH)P usually seen in the literature.*/
                auto I_KH = this->I - this->K * this->H;
                this->P = ((I_KH * this->P) * I_KH.transpose()) + ((this->K * this->R) * (this->K).transpose());
                // save the measurement and posterior state
                this->z = new_box;
                this->x_post = this->x;
                this->P_post = this->P;
                if (i != (time_gap - 1)) predict();
            }

            // todo: 虚拟轨迹和re-update还没写完呢？
            //            history_obs.pop_back();
            //            std::vector<Eigen::VectorXd *> box1 = *(--new_history.end()); // 倒数第二个
            //            std::vector<Eigen::VectorXd *> box2 = *(new_history.end()); // 非法内存访问！ 倒数第一个
        } /* if attr_saved is null, do nothing */
    }
}// namespace ocsort