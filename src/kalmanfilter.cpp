#include "../include/kalmanfilter.h"
#include <iostream>
namespace ocsort {
    KalmanFilterNew::KalmanFilterNew() {};
    KalmanFilterNew::KalmanFilterNew(int dim_x_, int dim_z_) {
        dim_x = dim_x_;
        dim_z = dim_z_;
        x = Eigen::VectorXf::Zero(dim_x_, 1);
        // fixme: Data type is double temporarily, P Q B F all initialized to identity matrix
        P = Eigen::MatrixXf::Identity(dim_x_, dim_x_);
        Q = Eigen::MatrixXf::Identity(dim_x_, dim_x_);
        B = Eigen::MatrixXf::Identity(dim_x_, dim_x_);// Generally not used
        F = Eigen::MatrixXf::Identity(dim_x_, dim_x_);
        H = Eigen::MatrixXf::Zero(dim_z_, dim_x_);
        R = Eigen::MatrixXf::Identity(dim_z_, dim_z_);
        M = Eigen::MatrixXf::Zero(dim_x_, dim_z_);
        z = Eigen::VectorXf::Zero(dim_z_, 1);
        /*
            gain and residual are computed during the innovation step. We
            save them so that in case you want to inspect them for various
            purposes
        * */
        K = Eigen::MatrixXf::Zero(dim_x_, dim_z_);
        y = Eigen::VectorXf::Zero(dim_x_, 1);
        S = Eigen::MatrixXf::Zero(dim_z_, dim_z_);
        // SI is transpose of S, why not directly S.T?

        // Below saves variables after predict
        x_prior = x;
        P_prior = P;
        // Below saves variables after update
        x_post = x;
        P_post = P;
    };
    void KalmanFilterNew::predict() {
        /**Predict next state (prior) using the Kalman filter state propagation
    equations. This function requires parameters, but in ocsort, no parameters are passed, so this is the parameter-less version, can be overloaded later
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
        // x = Fx+Bu , but here I don't consider u and B
        x = F * x;
        // P = FPF' + Q , where F' is transpose of F
        P.noalias() = _alpha_sq * ((F * P) * F.transpose()) + Q;
        // Save previous values
        x_prior = x;
        P_prior = P;
    }
    void KalmanFilterNew::update(Eigen::VectorXf z_) {
        // Function prototype: update(self, z, R=None, H=None), but R H are not used here
        // Passed parameters are pointer types to easily substitute None in python with null pointer
        /*
        Add a new measurement (z) to the Kalman filter.
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
        // NOTE: z_ shape is [dim_z,1] i.e. [4,1]
        // Add new observation to history_obs, array stores Eigen::VectorXf pointers
        history_obs.push_back(z_);
        if (z_.size() == 0) {// Indicates target lost
            if (true == observed) freeze();
            observed = false;// Added in ocsort
            // In original py code: z = np.array([ [None]*dim_z ]).T
            z = Eigen::VectorXf::Zero(dim_z, 1);
            x_post = x;
            P_post = P;
            y = Eigen::VectorXf::Zero(dim_z, 1);
            return;
        }
        // If observation received again, unfreeze
        /*
            Get observation, use online smoothing to re-update parameters => OOS
            Now this step is called ORU
            observed is False => Observation received. Unfreeze.
         */
        if (false == observed) unfreeze();
        observed = true;
        // Below is the formal update algorithm
        // y = z - Hx , residual between measurement and prediction
        y.noalias() = z_ - H * x;
        // Expression PH' is used in calculating S and K, so save it to reduce computation
        Eigen::MatrixXf PHT;
        PHT.noalias() = P * H.transpose();
        // S = HPH' + R, where H' is transpose of H, S is measurement residual covariance
        S.noalias() = H * PHT + R;
        // Inverse of S, used in subsequent calculations
        SI = S.inverse();
        // K = PH'SI , calculate optimal kalman gain
        K.noalias() = PHT * SI;
        // note: update x
        x.noalias() = x + K * y;
        // note: update P
        /*This is more numerically stable and works for non-optimal K vs the
         * equation P = (I-KH)P usually seen in the literature.*/
        Eigen::MatrixXf I_KH;
        I_KH.noalias() = I - K * H;
        Eigen::MatrixXf P_INT;
        P_INT.noalias() = (I_KH * P);
        P.noalias() = (P_INT * I_KH.transpose()) + ((K * R) * K.transpose());
        // save the measurement and posterior state
        z = z_;
        x_post = x;
        P_post = P;
    }
    void KalmanFilterNew::freeze() {
        /**
         * Save all variables in this object
         * save all the variable in current object at the time
         * */
        // note: Most important point, set flag to true first
        attr_saved.IsInitialized = true;
        /////////// Start saving ///////////
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
        /* Load all variables saved by freeze */
        // todo: Remember to add attr_saved.size > 0 condition later
        if (true == attr_saved.IsInitialized) {
            // std::cout << "Restoring trajectory" << std::endl;
            new_history = history_obs;
            /* Start data restoration */
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
            /* Data restoration complete */
            // Except the last obs, assign previous ones to history_obs, equivalent to removing the last observation in history_obs
            history_obs.erase(history_obs.end() - 1);
            // Because some parts in new_history might be nullptr, need to filter,
            // Here we only need to get the last two valid observations
            // box1 = new_history[index1] # Get the second to last real observation from history
            Eigen::VectorXf box1;           // Second to last
            Eigen::VectorXf box2;           // Last observation
            int lastNotNullIndex = -1;      // Index of last
            int secondLastNotNullIndex = -1;// Index of second to last
            for (int i = new_history.size() - 1; i >= 0; i--) {
                if (new_history[i].size() > 0) {// != nullptr) {
                    if (lastNotNullIndex == -1) {
                        // The current element is the last non -NULLPTR element
                        lastNotNullIndex = i;
                        box2 = new_history.at(lastNotNullIndex);
                    } else if (secondLastNotNullIndex == -1) {
                        // The current element is the last second non -NULLPTR element
                        secondLastNotNullIndex = i;
                        box1 = new_history.at(secondLastNotNullIndex);
                        break;
                    }
                }
            }
            // Calculate \Delta{t}
            double time_gap = lastNotNullIndex - secondLastNotNullIndex;
            // Extract data, calculate width and height
            double x1 = box1[0];
            double x2 = box2[0];
            double y1 = box1[1];
            double y2 = box2[1];
            double w1 = std::sqrt(box1[2] * box1[3]);
            double h1 = std::sqrt(box1[2] / box1[3]);
            double w2 = std::sqrt(box2[2] * box2[3]);
            double h2 = std::sqrt(box2[2] / box2[3]);
            // Calculate derivative with respect to time
            double dx = (x2 - x1) / time_gap;// Speed in X direction
            double dy = (y1 - y2) / time_gap;// Speed in Y direction
            double dw = (w2 - w1) / time_gap;// Rate of change of w
            double dh = (h2 - h1) / time_gap;// Rate of change of h

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
                Eigen::VectorXf new_box(4, 1);
                new_box << x, y, s, r;
                /*
                    I still use predict-update loop here to refresh the parameters,
                    but this can be faster by directly modifying the internal parameters
                    as suggested in the paper. I keep this naive but slow way for
                    easy read and understanding
                    NOTE: I will update parameters directly here.
                 */
                // Below is the formal update algorithm
                // y = z - Hx , residual between measurement and prediction
                this->y.noalias() = new_box - this->H * this->x;
                // Expression PH' is used in calculating S and K, so save it to reduce computation
                Eigen::MatrixXf PHT;
                PHT.noalias() = this->P * this->H.transpose();
                // S = HPH' + R, where H' is transpose of H, S is measurement residual covariance
                this->S.noalias() = this->H * PHT + this->R;
                // Inverse of S, used in subsequent calculations
                this->SI = (this->S).inverse();
                // K = PH'SI , calculate optimal kalman gain
                this->K.noalias() = PHT * this->SI;
                // note: update x
                this->x.noalias() = this->x + this->K * this->y;
                // note: update P
                /*This is more numerically stable and works for non-optimal K vs the
                 * equation P = (I-KH)P usually seen in the literature.*/
                Eigen::MatrixXf I_KH;
                I_KH.noalias() = this->I - this->K * this->H;
                Eigen::MatrixXf P_INT;
                P_INT.noalias() = (I_KH * this->P);
                this->P.noalias() = (P_INT * I_KH.transpose()) + ((this->K * this->R) * (this->K).transpose());
                // save the measurement and posterior state
                this->z = new_box;
                this->x_post = this->x;
                this->P_post = this->P;
                if (i != (time_gap - 1)) predict();
            }
        } /* if attr_saved is null, do nothing */
    }
}// namespace ocsort
