#include "yolo.h"
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "OCsort.h"
#include "cpu.h"
//#include <cstdio> /*为了使用内存安全的 snprintf 函数*/
#include <iostream>
#include <string>
#include <sstream>
/*************************************************************/
/************************* 工具函数区 *************************/
/*************************************************************/
static float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

static float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }
    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects) {
    if (faceobjects.empty())
        return;
    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked,
                              float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++) {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

static void
generate_grids_and_stride(const int target_w, const int target_h, std::vector<int> &strides,
                          std::vector<GridAndStride> &grid_strides) {
    for (int i = 0; i < (int) strides.size(); i++) {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++) {
            for (int g0 = 0; g0 < num_grid_w; g0++) {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat &pred,
                               float prob_threshold, std::vector<Object> &objects) {
    const int num_points = grid_strides.size();
    const int num_class = 80;
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++) {
        const float *scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++) {
            float confidence = scores[k];
            if (confidence > score) {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold) {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void *) pred.row(i));
            {
                ncnn::Layer *softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++) {
                float dis = 0.f;
                const float *dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++) {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;

            objects.push_back(obj);
        }
    }
}

/*************************************************************/
/************************* Yolo方法具体实现 *************************/
/*************************************************************/

Yolo::Yolo() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

/**
 * 用来加载Yolo模型
 * @param mgr
 * @param modeltype
 * @param _target_size
 * @param _mean_vals
 * @param _norm_vals
 * @param use_gpu
 * @return
 */
int Yolo::load(AAssetManager *mgr, const char *modeltype, int _target_size, const float *_mean_vals,
               const float *_norm_vals, bool use_gpu) {
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();
    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
    yolo.opt = ncnn::Option();
#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif
    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "yolov8%s.param", modeltype);
    sprintf(modelpath, "yolov8%s.bin", modeltype);

    yolo.load_param(mgr, parampath);
    yolo.load_model(mgr, modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

/**
 * 输入图片，输出检测结果
 * @param rgb
 * @param objects
 * @param prob_threshold
 * @param nms_threshold
 * @return
 */
int Yolo::detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold,
                 float nms_threshold) {
    int width = rgb.cols;
    int height = rgb.rows;
    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height,
                                                 w, h);
    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 0.f);
    in_pad.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("images", in_pad);
    std::vector<Object> proposals;
    ncnn::Mat out;
    ex.extract("output", out);
    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    generate_proposals(grid_strides, out, prob_threshold, proposals);
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);
    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];
        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
        // clip
        x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);
        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
    // sort objects by area
    struct {
        bool operator()(const Object &a, const Object &b) const {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);
    return 0;
}

// note: 创建一个追踪器(全局变量)
ocsort::OCSort tracker(0, 50, 1, 0.22136877277096445, 1, "giou", 0.3941737016672115, true);

Eigen::MatrixXf convertToEigen(const std::vector<Object> &objects) {
    // Create a matrix of size (n, 5) where n is the number of objects
    Eigen::MatrixXf objects_eigen(objects.size(), 6);
    for (size_t i = 0; i < objects.size(); ++i) {
        const Object &obj = objects[i];
        // Convert cv::Rect_<float> (x, y, width, height) to (x1, y1, x2, y2)
        float x1 = obj.rect.x;
        float y1 = obj.rect.y;
        float x2 = obj.rect.x + obj.rect.width;
        float y2 = obj.rect.y + obj.rect.height;
        // Set the values in the Eigen matrix (x1, y1, x2, y2, prob)
        objects_eigen(i, 0) = x1;
        objects_eigen(i, 1) = y1;
        objects_eigen(i, 2) = x2;
        objects_eigen(i, 3) = y2;
        objects_eigen(i, 4) = obj.prob;
        objects_eigen(i, 5) = obj.label;
    }
    return objects_eigen;
}

std::vector<float> convertToTLWH(const Eigen::RowVectorXf &input) {
    // 从输入中提取 x1, y1, x2, y2
    float x1 = input(0);
    float y1 = input(1);
    float x2 = input(2);
    float y2 = input(3);

    // 计算 tlwh 格式 [x, y, width, height]
    float width = x2 - x1;
    float height = y2 - y1;

    // 将结果存储在 std::vector<float> 中并返回
    return {x1, y1, width, height};
}

int Yolo::draw(cv::Mat &rgb, const std::vector<Object> &objects) {
    // 开始追踪，将结果保存在 output_stracks 中
    // note: 把objects转成 Eigen::MatrixXf
    Eigen::MatrixXf objects_eigen = convertToEigen(objects);
    std::vector<Eigen::RowVectorXf> output_stracks;
    if (!objects.empty())
        output_stracks = tracker.update(objects_eigen);
    else
        output_stracks = tracker.update(Eigen::MatrixXf(objects.size(), 6));

    int color_index = 0;

    // [Debug] 调试用
    if (output_stracks.size() != objects.size())
        __android_log_print(ANDROID_LOG_WARN, "??", "%s", "tracking objects number not match");

    for (size_t i = 0; i < output_stracks.size(); i++) {
        const Object &obj = objects[i];
        char TEXT[256];
        if (obj.label != 0) continue;
        // 标识追踪到的目标 注意到我们的输出是: `<x1>,<y1>,<x2>,<y2>,<ID>,<class>,<confidence>`, 所以第4个元素是UID
        sprintf(TEXT, "UID:%d %s", int(output_stracks.at(i)[4]), class_names[obj.label]);

        // 记录 tlwh(top-left,width-height) 的，方面Opencv直接画矩形
        std::vector<float> tlwh = convertToTLWH(output_stracks.at(i));
        // 为目标框分配颜色
        const unsigned char *color = colors[color_index % 19];
        color_index++;
        cv::Scalar cc(color[0], color[1], color[2]);
        // 绘制追踪到的目标的矩形
        cv::rectangle(rgb, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), cc, 2);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(TEXT, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                              1, &baseLine);
        int x = tlwh[0];
        int y = tlwh[1] - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;
        // 这个 rectangle 是为了label准备的
        cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                    cv::Size(label_size.width, label_size.height + baseLine)), cc,
                      -1);
        // label 的颜色
        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0)
                                                                    : cv::Scalar(255, 255, 255);
        // 绘制label文字
        cv::putText(rgb, TEXT, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    textcc, 1);
    }
    return 0;
}