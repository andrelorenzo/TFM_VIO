#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <vector>

#include <opencv2/core.hpp>

#include "config.hpp"
#include "feature_tracker.hpp"
#include "imu_vio.hpp"

namespace vo {

struct VisionFrame {
    cv::Mat bgr;
    cv::Mat gray;
    double timestamp_sec = 0.0;
    std::uint64_t frame_number = 0;
};

struct VioPoseOutput {
    bool initialized = false;
    bool valid = false;
    bool imu_used = false;
    bool scale_initialized = false;
    bool scale_updated = false;

    int frame_idx = 0;
    double timestamp_sec = 0.0;
    double frame_dt = 0.0;
    double fps = 0.0;

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    double x_visual_unscaled = 0.0;
    double y_visual_unscaled = 0.0;
    double z_visual_unscaled = 0.0;

    double x_imu = 0.0;
    double y_imu = 0.0;
    double z_imu = 0.0;

    double roll_deg = 0.0;
    double pitch_deg = 0.0;
    double yaw_deg = 0.0;

    double roll_deg_visual = 0.0;
    double pitch_deg_visual = 0.0;
    double yaw_deg_visual = 0.0;

    double roll_deg_imu = 0.0;
    double pitch_deg_imu = 0.0;
    double yaw_deg_imu = 0.0;

    cv::Vec4d quat_wxyz = cv::Vec4d(1.0, 0.0, 0.0, 0.0);

    int raw_matches = 0;
    int tracked_points = 0;
    int pose_inliers = 0;
    double median_flow_px = 0.0;
    double scale_estimate = 1.0;
};

class VisualInertialOdometry {
public:
    explicit VisualInertialOdometry(const Config& config);

    VioPoseOutput update(const VisionFrame& frame,
                         const ImuFrontendOutput* imu_state,
                         cv::Mat* debug_vis = nullptr);

    const VioPoseOutput& lastOutput() const { return last_output_; }
    bool isInitialized() const { return initialized_; }

private:
    struct CameraExtrinsics {
        cv::Mat R_ci = cv::Mat::eye(3, 3, CV_64F);  // imu -> color
        cv::Mat R_ic = cv::Mat::eye(3, 3, CV_64F);  // color -> imu
        cv::Vec3d t_ci = cv::Vec3d(0.0, 0.0, 0.0);  // origin of imu in color frame
        cv::Vec3d t_ic = cv::Vec3d(0.0, 0.0, 0.0);  // origin of color in imu frame
        bool valid = false;
    };

    struct ImuCameraPose {
        bool valid = false;
        double timestamp_sec = 0.0;
        cv::Mat R_wc = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat R_cw = cv::Mat::eye(3, 3, CV_64F);
        cv::Vec3d p_w = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d p_rel = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d rpy_deg = cv::Vec3d(0.0, 0.0, 0.0);
        double interval_dt = 0.0;
        cv::Vec3d preint_dtheta = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d preint_dv = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Vec3d preint_dp = cv::Vec3d(0.0, 0.0, 0.0);
    };

    struct ScaleWindowEntry {
        double timestamp_sec = 0.0;
        double visual_step = 0.0;
        double imu_step = 0.0;
    };

    struct RelativePoseEstimate {
        bool ok = false;
        TrackSet tracks;
        cv::Mat inlier_mask;
        cv::Mat R_21;  // prev/ref -> curr
        cv::Vec3d t_21 = cv::Vec3d(0.0, 0.0, 0.0);
        int pose_inliers = 0;
        double median_flow_px = 0.0;
    };

    struct Keyframe {
        cv::Mat gray;
        double timestamp_sec = 0.0;
        std::uint64_t frame_number = 0;
        cv::Vec3d visual_pos = cv::Vec3d(0.0, 0.0, 0.0);
        cv::Mat R_cw_ref = cv::Mat::eye(3, 3, CV_64F);
    };

    static cv::Mat identity_pose();
    static cv::Mat make_pose(const cv::Mat& R, const cv::Mat& t);
    static cv::Vec3d mat_to_vec3d(const cv::Mat& v);
    static cv::Mat vec3_to_mat(const cv::Vec3d& v);
    static cv::Vec3d rotation_matrix_to_rpy_deg(const cv::Mat& R_wc);
    static cv::Vec4d rotation_matrix_to_quat_wxyz(const cv::Mat& R_wc);
    static std::vector<cv::Point2f> undistort_to_normalized(
        const std::vector<cv::Point2f>& pts,
        const cv::Mat& K,
        const cv::Mat& dist);
    static double median_flow(
        const std::vector<cv::Point2f>& prev_pts,
        const std::vector<cv::Point2f>& curr_pts,
        const cv::Mat& mask);
    static void draw_tracks(
        const cv::Mat& bgr,
        const std::vector<cv::Point2f>& prev_pts,
        const std::vector<cv::Point2f>& curr_pts,
        const cv::Mat& mask,
        const VioPoseOutput& sample,
        cv::Mat& out);

    void validate_camera_model() const;
    void load_extrinsics();
    void update_fps();

    RelativePoseEstimate estimate_relative_pose(const cv::Mat& ref_gray, const cv::Mat& cur_gray) const;
    cv::Vec3d relative_motion_dir_world(const RelativePoseEstimate& estimate,
                                        const cv::Mat& R_cw_ref) const;
    bool try_relocalize_from_keyframes(const VisionFrame& frame,
                                       RelativePoseEstimate& estimate,
                                       cv::Vec3d& pos_candidate,
                                       cv::Mat* R_cw_candidate) const;
    bool should_add_keyframe(const VisionFrame& frame, const cv::Vec3d& pos_candidate) const;
    void maybe_add_keyframe(const VisionFrame& frame,
                            const cv::Vec3d& pos_candidate,
                            const cv::Mat& R_cw_ref);
    bool maybe_update_scale_estimate(double timestamp_sec,
                                     double frame_dt,
                                     const cv::Vec3d& visual_pos_current,
                                     const ImuCameraPose& imu_cam_pose);
    bool current_imu_camera_pose(const ImuFrontendOutput* imu_state,
                                 ImuCameraPose* pose) const;
    void reset_scale_tracking(const cv::Vec3d& visual_pos,
                              const ImuCameraPose& imu_pose);
    VioPoseOutput make_output(const VisionFrame& frame,
                              bool valid_update,
                              const ImuCameraPose& imu_pose,
                              const cv::Mat& R_cw_out,
                              const cv::Mat& R_cw_visual,
                              const RelativePoseEstimate& estimate) const;

    Config config_;
    OrbFlowTracker tracker_;
    CameraExtrinsics extrinsics_;

    cv::Mat T_cw_;
    cv::Mat R_cw_visual_;

    bool initialized_ = false;
    VisionFrame prev_frame_;

    cv::Vec3d visual_position_unscaled_ = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec3d imu_cam_origin_ = cv::Vec3d(0.0, 0.0, 0.0);
    bool imu_cam_origin_initialized_ = false;

    double global_scale_estimate_ = 1.0;
    bool global_scale_initialized_ = false;
    bool scale_updated_this_frame_ = false;
    cv::Vec3d scale_anchor_visual_pos_ = cv::Vec3d(0.0, 0.0, 0.0);
    cv::Vec3d scale_anchor_imu_pos_ = cv::Vec3d(0.0, 0.0, 0.0);
    double scale_anchor_timestamp_sec_ = 0.0;
    double scale_anchor_imu_timestamp_sec_ = 0.0;
    bool scale_anchor_initialized_ = false;

    std::deque<ScaleWindowEntry> scale_window_;
    std::deque<Keyframe> keyframes_;

    double first_ts_ = -1.0;
    double prev_frame_ts_ = -1.0;
    double fps_ema_ = 0.0;
    std::chrono::steady_clock::time_point last_tick_;

    VioPoseOutput last_output_;
};

} // namespace vo
