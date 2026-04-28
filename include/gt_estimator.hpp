#pragma once

#include <cstdint>
#include <vector>
#include <opencv2/core.hpp>


struct GroundTruthConfig
{
    int max_features = 2000;
    int min_points = 80;
    int min_inliers = 40;
    int lk_win_size = 21;
    int lk_max_level = 3;
    int depth_patch_radius = 1;
    bool enabled = false;
};

struct GroundTruthFrame
{
    cv::Mat rgb;
    cv::Mat gray;
    cv::Mat depth_m;     // CV_32F depth en metros, alineada con color
    cv::Mat K;           // 3x3 CV_64F
    cv::Mat dist;        // distorsion
    double timestamp_sec = 0.0;
    std::uint64_t frame_number = 0;
};

struct GroundTruthState
{
    bool initialized = false;
    bool valid = false;

    double timestamp_sec = 0.0;
    std::uint64_t frame_number = 0;

    cv::Vec3d xyz = {0.0, 0.0, 0.0};      // posicion camara en mundo
    cv::Vec3d rpy_rad = {0.0, 0.0, 0.0};  // orientacion camara en mundo
    cv::Vec3d rpy_deg = {0.0, 0.0, 0.0};
    cv::Vec4d quat = {1.0, 0.0, 0.0, 0.0};
    cv::Matx44d T_wc = cv::Matx44d::eye();

    int tracked_points = 0;
    int pnp_inliers = 0;
};

class GroundTruthEstimator
{
public:
    GroundTruthEstimator() = default;
    ~GroundTruthEstimator() = default;

    bool init(const GroundTruthConfig& cfg);
    void reset();
    void close();

    bool isEnabled() const { return cfg_.enabled; }
    bool isInitialized() const { return initialized_; }

    GroundTruthState update(const GroundTruthFrame& frame);

private:
    static double computeMedian(std::vector<float>& values);
    static cv::Mat identityPose();
    static cv::Mat makePose(const cv::Mat& R, const cv::Mat& t);
    static bool insideImage(const cv::Point2f& p, const cv::Size& size);
    static double sampleDepthPatch(const cv::Mat& depth_m, const cv::Point2f& px, int radius);
    static cv::Matx44d matToMatx44d(const cv::Mat& T);
    static cv::Vec4d rotationToQuaternion(const cv::Matx33d& R);
    static cv::Vec3d rotationToEulerRad(const cv::Matx33d& R);

private:
    GroundTruthConfig cfg_{};
    bool initialized_{false};
    bool has_prev_frame_{false};

    GroundTruthFrame prev_frame_;
    cv::Mat T_cw_;   // pose acumulada world->camera
};
