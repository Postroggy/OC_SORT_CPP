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
        KalmanFilterNew& operator=(const KalmanFilterNew&) = default;

    public:
        int dim_z = 4;
        int dim_x = 7;
        int dim_u = 0;
        // state 这是kalman状态变量 [7,1]
        Eigen::VectorXf x;
        // P是协方差矩阵，一开始先声明为单位矩阵，这里数据是float的。[7,7]
        Eigen::MatrixXf P;
        // Q是 过程噪音 协方差矩阵 [7,7]
        Eigen::MatrixXf Q;
        // B是 控制矩阵，实际上在目标追踪中都没用到过 [n,n]
        Eigen::MatrixXf B;
        // 预测矩阵\状态转移矩阵 [7,7]
        Eigen::Matrix<float,7,7> F;
        // 观测模型\矩阵 [4,7]
        Eigen::Matrix<float,4,7> H;
        // 观测噪音 [4,4]
        Eigen::Matrix<float,4,4> R;
        // fading memory control，控制更新权重float
        float _alpha_sq = 1.;
        // 测量矩阵，将状态向量x转换为测量向量z [7,4] 和矩阵H作用相反
        Eigen::MatrixXf M;
        // 测量向量 [4,1]
        Eigen::VectorXf z;
        /* 下面的变量是属于计算的中间变量 */
        // kalman 最优增益 [7,4]
        Eigen::MatrixXf K;
        // 测量残差 [4,1]
        Eigen::MatrixXf y;
        // 测量残差协方差
        Eigen::MatrixXf S;
        // 测量残差协方差的转置(化简后续运算)
        Eigen::MatrixXf SI;
        // 定义一个[dim_x,dim_x]的单位矩阵，方便后续运算，这个不能被更改
        const Eigen::MatrixXf I = Eigen::MatrixXf::Identity(dim_x, dim_x);
        // there will always be a copy of x,P after predict() is called
        // 如果需要两个Eigen矩阵之间相互赋值的话，前提是要初始化好，因为这样才知道col 和 row是否相等
        Eigen::VectorXf x_prior;
        Eigen::MatrixXf P_prior;
        // there will always be a copy of x,P after update() is called
        Eigen::VectorXf x_post;
        Eigen::MatrixXf P_post;
        // keeps all observation ,有z的时候，直接 push_back()
        std::vector<Eigen::VectorXf > history_obs;
        // 下面是 ocsort 新增加的
        // 用来标记追踪的状态(是否任然有目标与这个轨迹匹配),默认是 false
        bool observed = false;
        std::vector<Eigen::VectorXf > new_history; // 用于建立虚拟轨迹
        /* todo: 换一种方式存变量吧，C++中没有python的self.__dict__
         * 用 map<string,any>存内存开销大，而且重新赋值给Eigen数据的时候会报错，
         * 群里有人说用 metadata(元信息)可以做到，但是我不会
         * 所以这里用 结构体保存变量了
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
            std::vector<Eigen::VectorXf > history_obs;
            bool observed = false;
            // 下面是为了判断是否被freeze保存了数据
            bool IsInitialized = false;
        };
        struct Data attr_saved;
    };

}// namespace ocsort


#endif//OC_SORT_CPP_KALMANFILTER_H
