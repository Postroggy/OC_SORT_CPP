#ifndef OC_SORT_CPP_KALMANFILTER_H
#define OC_SORT_CPP_KALMANFILTER_H
#include <Eigen/Dense>
#include <any>
#include <map>
namespace ocsort {
    class KalmanFilterNew {
    public:
        KalmanFilterNew();
        KalmanFilterNew(int dim_x_, int dim_z_);
        void predict();
        void update(Eigen::VectorXf z_);
        void freeze();
        void unfreeze();
        KalmanFilterNew &operator=(const KalmanFilterNew &) = default;

    public:
        int dim_z = 4;
        int dim_x = 7;
        int dim_u = 0;
        // state This is kalman state variable [7,1]
        Eigen::VectorXf x;
        // P is covariance matrix, initially declared as identity matrix, data type is float. [7,7]
        Eigen::MatrixXf P;
        // Q is process noise covariance matrix [7,7]
        Eigen::MatrixXf Q;
        // B is control matrix, actually not used in object tracking [n,n]
        Eigen::MatrixXf B;
        // Prediction matrix / State transition matrix [7,7]
        Eigen::Matrix<float, 7, 7> F;
        // Observation model / matrix [4,7]
        Eigen::Matrix<float, 4, 7> H;
        // Observation noise [4,4]
        Eigen::Matrix<float, 4, 4> R;
        // fading memory control, controls update weight float
        float _alpha_sq = 1.;
        // Measurement matrix, converts state vector x to measurement vector z [7,4], opposite to matrix H
        Eigen::MatrixXf M;
        // Measurement vector [4,1]
        Eigen::VectorXf z;
        /* Variables below are intermediate variables for calculation */
        // kalman optimal gain [7,4]
        Eigen::MatrixXf K;
        // Measurement residual [4,1]
        Eigen::MatrixXf y;
        // Measurement residual covariance
        Eigen::MatrixXf S;
        // Transpose of measurement residual covariance (simplify subsequent operations)
        Eigen::MatrixXf SI;
        // Define a [dim_x,dim_x] identity matrix for subsequent operations, cannot be changed
        const Eigen::MatrixXf I = Eigen::MatrixXf::Identity(dim_x, dim_x);
        // there will always be a copy of x,P after predict() is called
        // If assigning between two Eigen matrices, they must be initialized first to ensure col and row match
        Eigen::VectorXf x_prior;
        Eigen::MatrixXf P_prior;
        // there will always be a copy of x,P after update() is called
        Eigen::VectorXf x_post;
        Eigen::MatrixXf P_post;
        // keeps all observation, push_back() directly when z is available
        std::vector<Eigen::VectorXf> history_obs;
        // Below are added in ocsort
        // Used to mark tracking state (whether there is still a target matching this trajectory), default is false
        bool observed = false;
        std::vector<Eigen::VectorXf> new_history;// Used to establish virtual trajectory
        /* todo: find another way to store variables, C++ doesn't have python's self.__dict__
         * using map<string,any> has high memory overhead, and errors when reassigning to Eigen data,
         * someone in the group said metadata could do it, but I don't know how
         * so using struct to save variables here
         * */
        struct Data {
            Eigen::VectorXf x;
            Eigen::MatrixXf P;
            Eigen::MatrixXf Q;
            Eigen::MatrixXf B;
            Eigen::MatrixXf F;
            Eigen::MatrixXf H;
            Eigen::MatrixXf R;
            float _alpha_sq = 1.;
            Eigen::MatrixXf M;
            Eigen::VectorXf z;
            Eigen::MatrixXf K;
            Eigen::MatrixXf y;
            Eigen::MatrixXf S;
            Eigen::MatrixXf SI;
            Eigen::VectorXf x_prior;
            Eigen::MatrixXf P_prior;
            Eigen::VectorXf x_post;
            Eigen::MatrixXf P_post;
            std::vector<Eigen::VectorXf> history_obs;
            bool observed = false;
            // Below is to judge whether data has been saved by freeze
            bool IsInitialized = false;
        };
        struct Data attr_saved;
    };

}// namespace ocsort


#endif//OC_SORT_CPP_KALMANFILTER_H
