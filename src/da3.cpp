#include "da3.h"

#include "preprocess.h"

#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

struct Da3Runtime {
    Config config;

    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::IoBinding> binding;

    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    size_t input_elements = 0;
    size_t output_elements = 0;
    int output_height = 0;
    int output_width = 0;

    Ort::MemoryInfo cuda_mem{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
    Ort::Value input_tensor{nullptr};
    Ort::Value output_tensor{nullptr};

    float* d_input = nullptr;
    float* d_output = nullptr;

    cv::cuda::GpuMat frame_gpu;
    cv::cuda::Stream stream;

    std::thread worker;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    cv::Mat pending_frame;
    bool has_pending = false;
    bool stop_requested = false;

    std::mutex result_mutex;
    EvitationDir latest_result;
    bool has_result = false;
    float filtered_dx = 0.0f;
    float filtered_dy = 0.0f;
    float filtered_score = 0.0f;
    bool active_state = false;
    cv::Mat latest_debug_image;

    bool initialized = false;
    bool enabled = false;
};

struct Da3DebugData {
    cv::Rect frontal_roi;
    cv::Rect guidance_roi;
    cv::Rect best_sector_roi;
    cv::Mat frontal_mask_full;
    cv::Point2f target_point = cv::Point2f(-1.0f, -1.0f);
    std::vector<float> sector_scores;
};

Da3Runtime g_da3;

std::wstring ToWideString(const std::string& input)
{
    return std::wstring(input.begin(), input.end());
}

size_t ShapeElements(const std::vector<int64_t>& shape)
{
    size_t elems = 1;
    for (int64_t dim : shape) {
        elems *= static_cast<size_t>(std::max<int64_t>(dim, 1));
    }
    return elems;
}

void FixInputShape(std::vector<int64_t>& shape, int input_height, int input_width)
{
    shape = {1, 3, input_height, input_width};
}

bool FixOutputShape(std::vector<int64_t>& shape, int input_height, int input_width, int& output_height, int& output_width)
{
    if (shape.empty()) {
        return false;
    }

    if (shape.size() >= 4) {
        if (shape[0] < 1) shape[0] = 1;
        if (shape[1] < 1) shape[1] = 1;
        if (shape[shape.size() - 2] < 1) shape[shape.size() - 2] = input_height;
        if (shape[shape.size() - 1] < 1) shape[shape.size() - 1] = input_width;
    } else if (shape.size() == 3) {
        if (shape[0] < 1) shape[0] = 1;
        if (shape[1] < 1) shape[1] = input_height;
        if (shape[2] < 1) shape[2] = input_width;
    } else if (shape.size() == 2) {
        if (shape[0] < 1) shape[0] = input_height;
        if (shape[1] < 1) shape[1] = input_width;
    } else {
        return false;
    }

    output_height = static_cast<int>(shape[shape.size() - 2]);
    output_width = static_cast<int>(shape[shape.size() - 1]);
    return output_height > 0 && output_width > 0;
}

void ReleaseInferenceResources()
{
    g_da3.binding.reset();
    g_da3.session.reset();
    g_da3.env.reset();
    g_da3.input_tensor = Ort::Value{nullptr};
    g_da3.output_tensor = Ort::Value{nullptr};

    if (g_da3.d_output != nullptr) {
        cudaFree(g_da3.d_output);
        g_da3.d_output = nullptr;
    }
    if (g_da3.d_input != nullptr) {
        cudaFree(g_da3.d_input);
        g_da3.d_input = nullptr;
    }

    g_da3.frame_gpu.release();
    g_da3.input_shape.clear();
    g_da3.output_shape.clear();
    g_da3.input_elements = 0;
    g_da3.output_elements = 0;
    g_da3.output_height = 0;
    g_da3.output_width = 0;
}

void PreprocessToGpu(const cv::cuda::GpuMat& frame_gpu, float* d_output, const Da3Config& config, cv::cuda::Stream& stream)
{
    cv::cuda::GpuMat resized;
    cv::cuda::GpuMat rgb;

    cv::cuda::resize(frame_gpu, resized, cv::Size(config.input_width, config.input_height), 0, 0, cv::INTER_LINEAR, stream);
    cv::cuda::cvtColor(resized, rgb, cv::COLOR_BGR2RGB, 0, stream);

    const cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
    LaunchRgb8ToChwF32Norm(rgb.ptr<unsigned char>(),rgb.step,config.input_height,config.input_width,d_output,kMean,kStd,cuda_stream);
}

cv::Mat ColorizeDepth(const cv::Mat& depth32f)
{
    double minv = 0.0;
    double maxv = 0.0;
    cv::minMaxLoc(depth32f, &minv, &maxv);

    cv::Mat norm8u;
    if (maxv - minv < 1e-6) {
        norm8u = cv::Mat::zeros(depth32f.size(), CV_8U);
    } else {
        cv::Mat norm01 = (depth32f - minv) / (maxv - minv);
        norm01.convertTo(norm8u, CV_8U, 255.0);
    }

    cv::Mat colored;
    cv::applyColorMap(norm8u, colored, cv::COLORMAP_INFERNO);
    return colored;
}

cv::Rect MakeCenteredRoi(const cv::Size& size, double width_ratio, double height_ratio)
{
    const int roi_w = std::max(1, std::min(size.width, static_cast<int>(std::lround(size.width * width_ratio))));
    const int roi_h = std::max(1, std::min(size.height, static_cast<int>(std::lround(size.height * height_ratio))));
    const int x = std::max(0, (size.width - roi_w) / 2);
    const int y = std::max(0, (size.height - roi_h) / 2);
    return cv::Rect(x, y, roi_w, roi_h);
}

float PercentileOfVector(std::vector<float> values, double q)
{
    if (values.empty()) {
        return 0.0f;
    }

    q = std::clamp(q, 0.0, 1.0);
    const std::size_t idx = static_cast<std::size_t>(std::floor(q * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + idx, values.end());
    return values[idx];
}

std::vector<float> CollectFiniteValues(const cv::Mat& img, float min_value, float max_value)
{
    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(img.total()));

    for (int y = 0; y < img.rows; ++y) {
        const float* row = img.ptr<float>(y);
        for (int x = 0; x < img.cols; ++x) {
            const float v = row[x];
            if (std::isfinite(v) && v >= min_value && v <= max_value) {
                values.push_back(v);
            }
        }
    }
    return values;
}

EvitationDir AnalyzeDepth(const cv::Mat& depth32f, Da3DebugData* debug)
{
    EvitationDir result;
    if (depth32f.empty() || depth32f.type() != CV_32F) {
        return result;
    }

    const Da3Config& cfg = g_da3.config.da3;
    const cv::Size size = depth32f.size();
    std::vector<float> global_depths = CollectFiniteValues(depth32f, static_cast<float>(cfg.min_valid_depth), static_cast<float>(cfg.max_valid_depth));
    if (global_depths.empty()) {
        return result;
    }

    result.valid_ratio = static_cast<float>(global_depths.size()) / static_cast<float>(depth32f.total());
    result.depth_p10 = PercentileOfVector(global_depths, 0.10);
    result.depth_p90 = PercentileOfVector(global_depths, 0.90);

    const float denom = std::max(1e-5f, result.depth_p90 - result.depth_p10);
    cv::Mat normalized_depth(depth32f.size(), CV_32F, cv::Scalar(0.0f));
    cv::Mat closeness(depth32f.size(), CV_32F, cv::Scalar(0.0f));

    for (int y = 0; y < depth32f.rows; ++y) {
        const float* src = depth32f.ptr<float>(y);
        float* dst_norm = normalized_depth.ptr<float>(y);
        float* dst_close = closeness.ptr<float>(y);
        for (int x = 0; x < depth32f.cols; ++x) {
            const float d = src[x];
            if (!std::isfinite(d) || d < cfg.min_valid_depth || d > cfg.max_valid_depth) {
                continue;
            }
            const float nd = std::clamp((d - result.depth_p10) / denom, 0.0f, 1.0f);
            dst_norm[x] = nd;
            dst_close[x] = 1.0f - nd;
        }
    }

    const cv::Rect frontal_roi = MakeCenteredRoi(size, cfg.frontal_roi_width_ratio, cfg.frontal_roi_height_ratio);
    const cv::Rect guidance_roi = MakeCenteredRoi(size, cfg.guidance_roi_width_ratio, cfg.guidance_roi_height_ratio);

    cv::Mat frontal_close = closeness(frontal_roi).clone();
    cv::Mat frontal_mask;
    cv::threshold(frontal_close, frontal_mask, cfg.close_mask_threshold, 255.0, cv::THRESH_BINARY);
    frontal_mask.convertTo(frontal_mask, CV_8U);

    const int morph_kernel = std::max(1, cfg.morph_kernel_size | 1);
    if (morph_kernel > 1) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(morph_kernel, morph_kernel));
        cv::morphologyEx(frontal_mask, frontal_mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(frontal_mask, frontal_mask, cv::MORPH_CLOSE, kernel);
    }

    result.mean_closeness = static_cast<float>(cv::mean(frontal_close)[0]);
    result.close_area_ratio = static_cast<float>(cv::countNonZero(frontal_mask)) / static_cast<float>(frontal_mask.total());

    std::vector<float> frontal_depths = CollectFiniteValues(depth32f(frontal_roi), static_cast<float>(cfg.min_valid_depth), static_cast<float>(cfg.max_valid_depth));
    if (!frontal_depths.empty()) {
        const double peak_percentile = std::clamp(cfg.frontal_peak_percentile, 0.0, 0.50);
        result.frontal_p20_depth = PercentileOfVector(frontal_depths, 0.20);
        result.frontal_peak_depth = PercentileOfVector(frontal_depths, peak_percentile);
        result.p20_closeness = std::clamp(1.0f - ((result.frontal_p20_depth - result.depth_p10) / denom), 0.0f, 1.0f);
        result.peak_closeness = std::clamp(1.0f - ((result.frontal_peak_depth - result.depth_p10) / denom), 0.0f, 1.0f);
    }

    int largest_blob_area = 0;
    if (cv::countNonZero(frontal_mask) > 0) {
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int n_labels = cv::connectedComponentsWithStats(frontal_mask, labels, stats, centroids, 8, CV_32S);
        for (int label = 1; label < n_labels; ++label) {
            largest_blob_area = std::max(largest_blob_area, stats.at<int>(label, cv::CC_STAT_AREA));
        }
    }
    result.largest_blob_ratio = static_cast<float>(largest_blob_area) / static_cast<float>(frontal_mask.total());

    const float raw_score =
        static_cast<float>(cfg.score_weight_mean) * result.mean_closeness +
        static_cast<float>(cfg.score_weight_area) * result.close_area_ratio +
        static_cast<float>(cfg.score_weight_blob) * result.largest_blob_ratio +
        static_cast<float>(cfg.score_weight_p20) * result.p20_closeness +
        static_cast<float>(cfg.score_weight_peak) * result.peak_closeness;

    const float alpha = static_cast<float>(std::clamp(cfg.smooth_alpha, 0.0, 1.0));
    if (!g_da3.has_result) {
        g_da3.filtered_score = raw_score;
    } else {
        g_da3.filtered_score = alpha * g_da3.filtered_score + (1.0f - alpha) * raw_score;
    }
    result.obstacle_score = g_da3.filtered_score;
    result.magnitude = result.obstacle_score;

    const bool depth_threshold_hit =
        cfg.evade_distance > 0.0 &&
        result.frontal_p20_depth > 0.0f &&
        result.frontal_p20_depth <= static_cast<float>(cfg.evade_distance);
    const bool depth_emergency_hit =
        cfg.evade_distance > 0.0 &&
        result.frontal_peak_depth > 0.0f &&
        result.frontal_peak_depth <= static_cast<float>(0.7 * cfg.evade_distance);
    const bool peak_hit = result.peak_closeness >= cfg.peak_score_threshold;
    const bool emergency_hit =
        depth_emergency_hit ||
        result.close_area_ratio >= cfg.emergency_close_area_ratio ||
        result.largest_blob_ratio >= cfg.emergency_blob_area_ratio;
    const bool threshold_hit =
        depth_threshold_hit ||
        peak_hit ||
        result.close_area_ratio >= cfg.min_close_area_ratio ||
        result.largest_blob_ratio >= cfg.min_blob_area_ratio ||
        result.obstacle_score >= cfg.activate_score_threshold;

    if (g_da3.active_state) {
        g_da3.active_state = emergency_hit || result.obstacle_score >= cfg.clear_score_threshold;
    } else {
        g_da3.active_state = emergency_hit || threshold_hit;
    }
    result.must_evade = g_da3.active_state;

    cv::Mat guidance_norm = normalized_depth(guidance_roi);
    cv::Mat guidance_close = closeness(guidance_roi);
    const int sector_count = std::max(3, cfg.sector_count);
    const int sector_width = std::max(1, guidance_roi.width / sector_count);

    float best_sector_score = -std::numeric_limits<float>::infinity();
    cv::Rect best_sector_local(0, 0, guidance_roi.width, guidance_roi.height);
    cv::Point2f target_point(static_cast<float>(size.width) * 0.5f, static_cast<float>(size.height) * 0.5f);

    if (debug != nullptr) {
        debug->sector_scores.clear();
        debug->sector_scores.reserve(static_cast<std::size_t>(sector_count));
    }

    for (int i = 0; i < sector_count; ++i) {
        const int x0 = i * sector_width;
        const int width = (i == sector_count - 1) ? (guidance_roi.width - x0) : sector_width;
        cv::Rect sector_local(x0, 0, std::max(1, width), guidance_roi.height);
        cv::Mat sector_norm = guidance_norm(sector_local);
        cv::Mat sector_close = guidance_close(sector_local);

        const float mean_free = static_cast<float>(cv::mean(sector_norm)[0]);
        cv::Mat sector_mask;
        cv::threshold(sector_close, sector_mask, cfg.close_mask_threshold, 255.0, cv::THRESH_BINARY);
        sector_mask.convertTo(sector_mask, CV_8U);
        const float sector_close_ratio = static_cast<float>(cv::countNonZero(sector_mask)) / static_cast<float>(sector_mask.total());
        const float sector_score = mean_free - static_cast<float>(cfg.sector_close_penalty) * sector_close_ratio;

        if (debug != nullptr) {
            debug->sector_scores.push_back(sector_score);
        }

        if (sector_score > best_sector_score) {
            best_sector_score = sector_score;
            best_sector_local = sector_local;
        }
    }

    result.free_space_score = best_sector_score;
    result.best_sector = std::clamp(best_sector_local.x / sector_width, 0, sector_count - 1);

    {
        cv::Mat best_sector_norm = guidance_norm(best_sector_local);
        double sum_w = 0.0;
        double sum_x = 0.0;
        double sum_y = 0.0;

        for (int y = 0; y < best_sector_norm.rows; ++y) {
            const float* row = best_sector_norm.ptr<float>(y);
            for (int x = 0; x < best_sector_norm.cols; ++x) {
                const double w = std::pow(static_cast<double>(row[x]), cfg.free_space_power);
                if (w <= 1e-6) continue;
                sum_w += w;
                sum_x += w * static_cast<double>(x + best_sector_local.x + guidance_roi.x);
                sum_y += w * static_cast<double>(y + best_sector_local.y + guidance_roi.y);
            }
        }

        if (sum_w > 1e-6) {
            target_point.x = static_cast<float>(sum_x / sum_w);
            target_point.y = static_cast<float>(sum_y / sum_w);
        } else {
            target_point.x = static_cast<float>(guidance_roi.x + best_sector_local.x + best_sector_local.width / 2);
            target_point.y = static_cast<float>(guidance_roi.y + best_sector_local.height / 2);
        }
    }

    const cv::Point2f image_center(static_cast<float>(size.width) * 0.5f, static_cast<float>(size.height) * 0.5f);
    const float raw_dx = target_point.x - image_center.x;
    const float raw_dy = target_point.y - image_center.y;
    const float raw_norm = std::sqrt(raw_dx * raw_dx + raw_dy * raw_dy);

    float unit_dx = 0.0f;
    float unit_dy = 0.0f;
    if (raw_norm > 1e-6f) {
        unit_dx = raw_dx / raw_norm;
        unit_dy = raw_dy / raw_norm;
    }

    if (!g_da3.has_result) {
        g_da3.filtered_dx = unit_dx;
        g_da3.filtered_dy = unit_dy;
    } else {
        g_da3.filtered_dx = alpha * g_da3.filtered_dx + (1.0f - alpha) * unit_dx;
        g_da3.filtered_dy = alpha * g_da3.filtered_dy + (1.0f - alpha) * unit_dy;
    }

    const float filtered_norm = std::sqrt(g_da3.filtered_dx * g_da3.filtered_dx + g_da3.filtered_dy * g_da3.filtered_dy);
    if (filtered_norm > cfg.direction_deadband) {
        result.norm_vec = vec3(g_da3.filtered_dx / filtered_norm, g_da3.filtered_dy / filtered_norm, 0.0);
    }

    if (debug != nullptr) {
        debug->frontal_roi = frontal_roi;
        debug->guidance_roi = guidance_roi;
        debug->best_sector_roi = cv::Rect(
            guidance_roi.x + best_sector_local.x,
            guidance_roi.y + best_sector_local.y,
            best_sector_local.width,
            best_sector_local.height
        );
        debug->frontal_mask_full = cv::Mat::zeros(size, CV_8U);
        frontal_mask.copyTo(debug->frontal_mask_full(frontal_roi));
        debug->target_point = target_point;
    }

    return result;
}

void StoreDebugImage(const cv::Mat& depth32f, const EvitationDir& result, const Da3DebugData& dbg)
{
    cv::Mat debug = ColorizeDepth(depth32f);
    if (debug.empty()) {
        return;
    }

    // Mask
    if (!dbg.frontal_mask_full.empty()) {
        cv::Mat mask_color(debug.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        mask_color.setTo(cv::Scalar(0, 255, 0), dbg.frontal_mask_full > 0);
        cv::addWeighted(debug, 1.0, mask_color, 0.25, 0.0, debug);
    }


    // ROI F, G and S*
    cv::rectangle(debug, dbg.guidance_roi, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    cv::rectangle(debug, dbg.frontal_roi, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    cv::rectangle(debug, dbg.best_sector_roi, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

    const cv::Point center(debug.cols / 2, debug.rows / 2);
    cv::circle(debug, center, 4, cv::Scalar(255, 255, 255), -1, cv::LINE_AA);
    if (dbg.target_point.x >= 0.0f && dbg.target_point.y >= 0.0f) {
        cv::circle(debug, dbg.target_point, 5, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
    }

    const float max_len = 0.35f * static_cast<float>(std::min(debug.cols, debug.rows));
    const float draw_len = std::clamp(result.magnitude * static_cast<float>(g_da3.config.da3.debug_arrow_gain), 0.0f, max_len);
    const cv::Scalar arrow_color = result.must_evade ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    if (draw_len > 1.0f) {
        const cv::Point tip(
            static_cast<int>(std::lround(center.x + result.norm_vec.x() * draw_len)),
            static_cast<int>(std::lround(center.y + result.norm_vec.y() * draw_len))
        );
        cv::arrowedLine(debug, center, tip, arrow_color, 2, cv::LINE_AA, 0, 0.2);
    }

    const Da3Config& cfg = g_da3.config.da3;

    const double font_scale = 0.35;
    const int font_thickness = 1;
    const int line_height = 13;
    const int text_x = 5;
    int text_y = 14;

    auto drawConditionLine = [&](const std::string& text, bool condition_ok) {
        const cv::Scalar color = condition_ok ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(debug, text, cv::Point(text_x, text_y), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        cv::putText(debug, text, cv::Point(text_x, text_y), cv::FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv::LINE_AA);
        text_y += line_height;
    };

    auto makeConditionText = [&](const std::string& name, float value, const std::string& op, float threshold) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3) << name << ": " << value << " " << op << " " << threshold;
        return ss.str();
    };

    // const bool cond_depth = result.frontal_p20_depth > 0.0f && result.frontal_p20_depth <= static_cast<float>(cfg.evade_distance);
    // const bool cond_peak = result.peak_closeness >= static_cast<float>(cfg.peak_score_threshold);
    // const bool cond_area = result.close_area_ratio >= static_cast<float>(cfg.min_close_area_ratio);
    // const bool cond_blob = result.largest_blob_ratio >= static_cast<float>(cfg.min_blob_area_ratio);
    // const bool cond_score_on = result.obstacle_score >= static_cast<float>(cfg.activate_score_threshold);
    // const bool cond_score_off = result.obstacle_score >= static_cast<float>(cfg.clear_score_threshold);

    drawConditionLine("must_evade: " + std::string(result.must_evade ? "TRUE" : "FALSE"), result.must_evade);
    // drawConditionLine(makeConditionText("d20", result.frontal_p20_depth, "<=", static_cast<float>(cfg.evade_distance)), cond_depth);
    // drawConditionLine(makeConditionText("Cp", result.peak_closeness, ">=", static_cast<float>(cfg.peak_score_threshold)), cond_peak);
    // drawConditionLine(makeConditionText("Af", result.close_area_ratio, ">=", static_cast<float>(cfg.min_close_area_ratio)), cond_area);
    // drawConditionLine(makeConditionText("Bf", result.largest_blob_ratio, ">=", static_cast<float>(cfg.min_blob_area_ratio)), cond_blob);
    // drawConditionLine(makeConditionText("S on", result.obstacle_score, ">=", static_cast<float>(cfg.activate_score_threshold)), cond_score_on);
    // drawConditionLine(makeConditionText("S off", result.obstacle_score, ">=", static_cast<float>(cfg.clear_score_threshold)), cond_score_off);
    std::lock_guard<std::mutex> lock(g_da3.result_mutex);
    g_da3.latest_debug_image = debug;
}

bool RunInference(const cv::Mat& frame_cpu)
{
    if (frame_cpu.empty() || frame_cpu.type() != CV_8UC3) {
        return false;
    }

    g_da3.frame_gpu.upload(frame_cpu, g_da3.stream);
    PreprocessToGpu(g_da3.frame_gpu, g_da3.d_input, g_da3.config.da3, g_da3.stream);
    g_da3.stream.waitForCompletion();

    g_da3.session->Run(Ort::RunOptions{nullptr}, *g_da3.binding);

    std::vector<float> depth_host(g_da3.output_elements);
    cudaMemcpyAsync(
        depth_host.data(),
        g_da3.d_output,
        g_da3.output_elements * sizeof(float),
        cudaMemcpyDeviceToHost,
        cv::cuda::StreamAccessor::getStream(g_da3.stream)
    );
    g_da3.stream.waitForCompletion();

    cv::Mat depth(g_da3.output_height, g_da3.output_width, CV_32F, depth_host.data());
    Da3DebugData dbg;
    EvitationDir result = AnalyzeDepth(depth, &dbg);

    {
        std::lock_guard<std::mutex> lock(g_da3.result_mutex);
        g_da3.latest_result = result;
        g_da3.has_result = true;
    }

    if (g_da3.config.da3.show_window) {
        StoreDebugImage(depth.clone(), result, dbg);
    }
    return true;
}

void WorkerLoop()
{
    while (true) {
        cv::Mat frame_cpu;

        {
            std::unique_lock<std::mutex> lock(g_da3.queue_mutex);
            g_da3.queue_cv.wait(lock, [] { return g_da3.stop_requested || g_da3.has_pending; });

            if (g_da3.stop_requested && !g_da3.has_pending) {
                break;
            }

            frame_cpu = g_da3.pending_frame;
            g_da3.pending_frame.release();
            g_da3.has_pending = false;
        }

        try {
            RunInference(frame_cpu);
        } catch (const std::exception& e) {
            Logger(ERROR, "da3 worker failed: %s", e.what());
        }
    }
}

bool InitInference()
{
    const Da3Config& config = g_da3.config.da3;
    int cuda_device_count = 0;
    if (cudaGetDeviceCount(&cuda_device_count) != cudaSuccess || cuda_device_count <= 0) {
        Logger(ERROR, "da3 init failed: no CUDA device available");
        return false;
    }

    const int device_id = 0;
    if (cudaSetDevice(device_id) != cudaSuccess) {
        Logger(ERROR, "da3 init failed: cudaSetDevice(%d) failed", device_id);
        return false;
    }

    cv::cuda::setDevice(device_id);

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        Logger(ERROR, "da3 init failed: could not query CUDA device properties");
        return false;
    }

    Logger(INFO,
           "da3Init: using CUDA device %d/%d => %s | cc=%d.%d | vram=%.2f GiB",
           device_id,
           cuda_device_count,
           prop.name,
           prop.major,
           prop.minor,
           static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));

    g_da3.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "da3");

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = device_id;
    cuda_options.arena_extend_strategy = 0;
    cuda_options.gpu_mem_limit = std::numeric_limits<size_t>::max();
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.do_copy_in_default_stream = 1;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    const std::wstring model_path = ToWideString(config.model_path);
    g_da3.session = std::make_unique<Ort::Session>(*g_da3.env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = g_da3.session->GetInputNameAllocated(0, allocator);
    auto output_name = g_da3.session->GetOutputNameAllocated(0, allocator);
    g_da3.input_name = input_name.get();
    g_da3.output_name = output_name.get();

    FixInputShape(g_da3.input_shape, config.input_height, config.input_width);
    g_da3.input_elements = ShapeElements(g_da3.input_shape);

    auto output_info = g_da3.session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    g_da3.output_shape = output_info.GetShape();
    if (!FixOutputShape(g_da3.output_shape, config.input_height, config.input_width, g_da3.output_height, g_da3.output_width)) {
        Logger(ERROR, "da3 init failed: unsupported output tensor rank");
        return false;
    }
    g_da3.output_elements = ShapeElements(g_da3.output_shape);

    if (cudaMalloc(reinterpret_cast<void**>(&g_da3.d_input), g_da3.input_elements * sizeof(float)) != cudaSuccess) {
        Logger(ERROR, "da3 init failed: could not allocate GPU input buffer");
        return false;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&g_da3.d_output), g_da3.output_elements * sizeof(float)) != cudaSuccess) {
        Logger(ERROR, "da3 init failed: could not allocate GPU output buffer");
        return false;
    }

    g_da3.input_tensor = Ort::Value::CreateTensor<float>(
        g_da3.cuda_mem,
        g_da3.d_input,
        g_da3.input_elements,
        g_da3.input_shape.data(),
        g_da3.input_shape.size()
    );
    g_da3.output_tensor = Ort::Value::CreateTensor<float>(
        g_da3.cuda_mem,
        g_da3.d_output,
        g_da3.output_elements,
        g_da3.output_shape.data(),
        g_da3.output_shape.size()
    );

    g_da3.binding = std::make_unique<Ort::IoBinding>(*g_da3.session);
    g_da3.binding->BindInput(g_da3.input_name.c_str(), g_da3.input_tensor);
    g_da3.binding->BindOutput(g_da3.output_name.c_str(), g_da3.output_tensor);

    Logger(INFO,
           "da3Init: input=%dx%d output=%dx%d model_path=%s",
           config.input_width,
           config.input_height,
           g_da3.output_width,
           g_da3.output_height,
           config.model_path.c_str());
    return true;
}

} // namespace

void da3Init(const Config* config)
{
    da3Close();

    if (config == nullptr) {
        Logger(WARN, "da3Init: null config");
        return;
    }

    g_da3.config = *config;
    g_da3.enabled = config->da3.enabled && config->gen.color_on;

    if (!g_da3.enabled) {
        Logger(INFO, "da3Init: module disabled");
        return;
    }

    try {
        if (!InitInference()) {
            ReleaseInferenceResources();
            return;
        }

        g_da3.stop_requested = false;
        g_da3.initialized = true;
        g_da3.worker = std::thread(WorkerLoop);
    } catch (const std::exception& e) {
        Logger(ERROR,
               "da3Init failed: %s. Check that ONNX Runtime CUDA DLLs are next to the executable, especially "
               "onnxruntime.dll, onnxruntime_providers_shared.dll and onnxruntime_providers_cuda.dll.",
               e.what());
        ReleaseInferenceResources();
        g_da3.initialized = false;
        g_da3.enabled = false;
    }
}

void da3Update(const SourceIn* source)
{
    if (!g_da3.initialized || source == nullptr || source->frame.empty()) {
        return;
    }

    if (source->frame.type() != CV_8UC3) {
        return;
    }

    cv::Mat frame_copy = source->frame.clone();
    {
        std::lock_guard<std::mutex> lock(g_da3.queue_mutex);
        g_da3.pending_frame = std::move(frame_copy);
        g_da3.has_pending = true;
    }
    g_da3.queue_cv.notify_one();
}

EvitationDir da3Get()
{
    std::lock_guard<std::mutex> lock(g_da3.result_mutex);
    return g_da3.latest_result;
}

cv::Mat da3GetDebugImage()
{
    std::lock_guard<std::mutex> lock(g_da3.result_mutex);
    return g_da3.latest_debug_image.clone();
}

void da3Close()
{
    {
        std::lock_guard<std::mutex> lock(g_da3.queue_mutex);
        g_da3.stop_requested = true;
    }
    g_da3.queue_cv.notify_one();

    if (g_da3.worker.joinable()) {
        g_da3.worker.join();
    }

    {
        std::lock_guard<std::mutex> lock(g_da3.queue_mutex);
        g_da3.pending_frame.release();
        g_da3.has_pending = false;
        g_da3.stop_requested = false;
    }

    ReleaseInferenceResources();

    {
        std::lock_guard<std::mutex> lock(g_da3.result_mutex);
        g_da3.latest_result = EvitationDir{};
        g_da3.latest_debug_image.release();
        g_da3.has_result = false;
        g_da3.filtered_dx = 0.0f;
        g_da3.filtered_dy = 0.0f;
        g_da3.filtered_score = 0.0f;
        g_da3.active_state = false;
    }

    g_da3.initialized = false;
    g_da3.enabled = false;
}
