#include "visual_odometry.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace vo {
namespace {

bool mask_at(const cv::Mat& mask, std::size_t idx) {
    if (mask.empty()) {
        return true;
    }
    if (mask.rows == 1) {
        return mask.at<uchar>(0, static_cast<int>(idx)) != 0;
    }
    return mask.at<uchar>(static_cast<int>(idx), 0) != 0;
}

cv::Vec3d normalized_or_zero(const cv::Vec3d& v) {
    const double n = cv::norm(v);
    if (n <= 1e-12 || !std::isfinite(n)) {
        return cv::Vec3d(0.0, 0.0, 0.0);
    }
    return v * (1.0 / n);
}

double clamp01(double x) {
    return std::max(0.0, std::min(1.0, x));
}

double robust_median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }

    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<long>(mid), values.end());
    double median = values[mid];

    if ((values.size() % 2) == 0) {
        const auto max_it = std::max_element(values.begin(), values.begin() + static_cast<long>(mid));
        median = 0.5 * (median + *max_it);
    }
    return median;
}

cv::Mat quat_wxyz_to_R(const cv::Vec4d& q_in) {
    double w = q_in[0];
    double x = q_in[1];
    double y = q_in[2];
    double z = q_in[3];

    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    if (n <= 1e-12 || !std::isfinite(n)) {
        return cv::Mat::eye(3, 3, CV_64F);
    }

    w /= n;
    x /= n;
    y /= n;
    z /= n;

    return (cv::Mat_<double>(3, 3)
        << 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
           2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
           2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y));
}

} // namespace

VisualInertialOdometry::VisualInertialOdometry(const Config& config)
    : config_(config),
      tracker_(config),
      T_cw_(identity_pose()),
      R_cw_visual_(cv::Mat::eye(3, 3, CV_64F)),
      global_scale_estimate_(1.0),
      last_tick_(std::chrono::steady_clock::now()) {
    validate_camera_model();
    load_extrinsics();
}

cv::Mat VisualInertialOdometry::identity_pose() {
    return cv::Mat::eye(4, 4, CV_64F);
}

cv::Mat VisualInertialOdometry::make_pose(const cv::Mat& R, const cv::Mat& t) {
    cv::Mat T = identity_pose();
    R.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
    t.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));
    return T;
}

cv::Vec3d VisualInertialOdometry::mat_to_vec3d(const cv::Mat& v) {
    cv::Mat tmp;
    v.convertTo(tmp, CV_64F);
    if (tmp.rows == 3 && tmp.cols == 1) {
        return cv::Vec3d(tmp.at<double>(0, 0), tmp.at<double>(1, 0), tmp.at<double>(2, 0));
    }
    if (tmp.rows == 1 && tmp.cols == 3) {
        return cv::Vec3d(tmp.at<double>(0, 0), tmp.at<double>(0, 1), tmp.at<double>(0, 2));
    }
    throw std::runtime_error("Expected a 3x1 or 1x3 matrix.");
}

cv::Mat VisualInertialOdometry::vec3_to_mat(const cv::Vec3d& v) {
    return (cv::Mat_<double>(3, 1) << v[0], v[1], v[2]);
}

cv::Vec3d VisualInertialOdometry::rotation_matrix_to_rpy_deg(const cv::Mat& R_wc) {
    cv::Mat R;
    R_wc.convertTo(R, CV_64F);

    const double roll = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
    const double pitch = std::asin(-std::max(-1.0, std::min(1.0, R.at<double>(2, 0))));
    const double yaw = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));

    const double rad_to_deg = 180.0 / 3.14159265358979323846;
    return cv::Vec3d(roll * rad_to_deg, pitch * rad_to_deg, yaw * rad_to_deg);
}

cv::Vec4d VisualInertialOdometry::rotation_matrix_to_quat_wxyz(const cv::Mat& R_wc) {
    cv::Mat R;
    R_wc.convertTo(R, CV_64F);

    cv::Matx33d M(
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2));

    cv::Vec4d q;
    const double trace = M(0, 0) + M(1, 1) + M(2, 2);
    if (trace > 0.0) {
        const double s = 0.5 / std::sqrt(trace + 1.0);
        q[0] = 0.25 / s;
        q[1] = (M(2, 1) - M(1, 2)) * s;
        q[2] = (M(0, 2) - M(2, 0)) * s;
        q[3] = (M(1, 0) - M(0, 1)) * s;
    } else if (M(0, 0) > M(1, 1) && M(0, 0) > M(2, 2)) {
        const double s = 2.0 * std::sqrt(1.0 + M(0, 0) - M(1, 1) - M(2, 2));
        q[0] = (M(2, 1) - M(1, 2)) / s;
        q[1] = 0.25 * s;
        q[2] = (M(0, 1) + M(1, 0)) / s;
        q[3] = (M(0, 2) + M(2, 0)) / s;
    } else if (M(1, 1) > M(2, 2)) {
        const double s = 2.0 * std::sqrt(1.0 + M(1, 1) - M(0, 0) - M(2, 2));
        q[0] = (M(0, 2) - M(2, 0)) / s;
        q[1] = (M(0, 1) + M(1, 0)) / s;
        q[2] = 0.25 * s;
        q[3] = (M(1, 2) + M(2, 1)) / s;
    } else {
        const double s = 2.0 * std::sqrt(1.0 + M(2, 2) - M(0, 0) - M(1, 1));
        q[0] = (M(1, 0) - M(0, 1)) / s;
        q[1] = (M(0, 2) + M(2, 0)) / s;
        q[2] = (M(1, 2) + M(2, 1)) / s;
        q[3] = 0.25 * s;
    }

    const double n = std::sqrt(q.dot(q));
    if (n <= 1e-12 || !std::isfinite(n)) {
        return cv::Vec4d(1.0, 0.0, 0.0, 0.0);
    }
    return q * (1.0 / n);
}

std::vector<cv::Point2f> VisualInertialOdometry::undistort_to_normalized(
    const std::vector<cv::Point2f>& pts,
    const cv::Mat& K,
    const cv::Mat& dist) {
    std::vector<cv::Point2f> out;
    if (pts.empty()) {
        return out;
    }
    cv::undistortPoints(pts, out, K, dist);
    return out;
}

double VisualInertialOdometry::median_flow(const std::vector<cv::Point2f>& prev_pts,
                                           const std::vector<cv::Point2f>& curr_pts,
                                           const cv::Mat& mask) {
    std::vector<double> values;
    values.reserve(prev_pts.size());

    for (std::size_t i = 0; i < prev_pts.size() && i < curr_pts.size(); ++i) {
        if (!mask_at(mask, i)) {
            continue;
        }
        values.push_back(cv::norm(curr_pts[i] - prev_pts[i]));
    }

    if (values.empty()) {
        return 0.0;
    }

    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<long>(mid), values.end());
    return values[mid];
}

void VisualInertialOdometry::draw_tracks(
    const cv::Mat& bgr,
    const std::vector<cv::Point2f>& prev_pts,
    const std::vector<cv::Point2f>& curr_pts,
    const cv::Mat& mask,
    const VioPoseOutput& sample,
    cv::Mat& out) {
    if (bgr.empty()) {
        out.release();
        return;
    }

    out = bgr.clone();

    for (std::size_t i = 0; i < prev_pts.size() && i < curr_pts.size(); ++i) {
        const bool inlier = mask_at(mask, i);
        const cv::Scalar color = inlier ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::line(out, prev_pts[i], curr_pts[i], color, 1, cv::LINE_AA);
        cv::circle(out, curr_pts[i], 2, color, -1, cv::LINE_AA);
    }

    std::ostringstream oss1;
    oss1 << "xyz = [" << std::fixed << std::setprecision(3)
         << sample.x << ", " << sample.y << ", " << sample.z << "] m";
    std::ostringstream oss2;
    oss2 << "vis unscaled = [" << std::fixed << std::setprecision(3)
         << sample.x_visual_unscaled << ", " << sample.y_visual_unscaled << ", " << sample.z_visual_unscaled << "]";
    std::ostringstream oss3;
    oss3 << "matches=" << sample.raw_matches
         << " tracked=" << sample.tracked_points
         << " inliers=" << sample.pose_inliers
         << " flow=" << std::fixed << std::setprecision(2) << sample.median_flow_px << " px";
    std::ostringstream oss4;
    oss4 << "scale=" << std::fixed << std::setprecision(3) << sample.scale_estimate
         << (sample.scale_initialized ? "" : " (warmup)");
    std::ostringstream oss5;
    oss5 << "rpy = [" << std::fixed << std::setprecision(1)
         << sample.roll_deg << ", " << sample.pitch_deg << ", " << sample.yaw_deg << "] deg";

    cv::putText(out, oss1.str(), cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(out, oss2.str(), cv::Point(12, 48), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(out, oss3.str(), cv::Point(12, 72), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(out, oss4.str(), cv::Point(12, 96), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::putText(out, oss5.str(), cv::Point(12, 120), cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    if (!sample.valid) {
        cv::putText(out, "visual update skipped / relocalization pending",
                    cv::Point(12, 144), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
}

void VisualInertialOdometry::validate_camera_model() const {
    if (config_.cam.w <= 0 || config_.cam.h <= 0) {
        throw std::runtime_error("Camera model invalido: width/height deben ser positivos.");
    }
    if (config_.cam.K.empty() || config_.cam.K.rows != 3 || config_.cam.K.cols != 3) {
        throw std::runtime_error("Camera model invalido: K debe ser 3x3.");
    }
    if (config_.cam.D.empty()) {
        throw std::runtime_error("Camera model invalido: faltan coeficientes de distorsion.");
    }
}

void VisualInertialOdometry::load_extrinsics() {
    extrinsics_ = CameraExtrinsics{};

    if (config_.imu.T.empty()) {
        return;
    }
    if (config_.imu.T.rows < 3 || config_.imu.T.cols < 4) {
        throw std::runtime_error("imu.tocolor debe ser al menos 3x4 (o 4x4).");
    }

    cv::Mat T;
    config_.imu.T.convertTo(T, CV_64F);
    extrinsics_.R_ci = T(cv::Range(0, 3), cv::Range(0, 3)).clone();
    extrinsics_.R_ic = extrinsics_.R_ci.t();

    const cv::Vec3d t_ci(
        T.at<double>(0, 3),
        T.at<double>(1, 3),
        T.at<double>(2, 3));
    extrinsics_.t_ci = t_ci;
    extrinsics_.t_ic = mat_to_vec3d(-extrinsics_.R_ic * vec3_to_mat(t_ci));
    extrinsics_.valid = true;
}

void VisualInertialOdometry::update_fps() {
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - last_tick_).count();
    last_tick_ = now;

    if (dt <= 0.0) {
        return;
    }

    const double inst_fps = 1.0 / dt;
    if (fps_ema_ <= 0.0) {
        fps_ema_ = inst_fps;
    } else {
        fps_ema_ = 0.9 * fps_ema_ + 0.1 * inst_fps;
    }
}

VisualInertialOdometry::RelativePoseEstimate VisualInertialOdometry::estimate_relative_pose(const cv::Mat& ref_gray, const cv::Mat& cur_gray) const {
    RelativePoseEstimate out;

    TrackSet tracks;
    if (!tracker_.track(ref_gray, cur_gray, tracks)) {
        out.tracks = std::move(tracks);
        return out;
    }

    out.tracks = tracks;
    out.median_flow_px = median_flow(tracks.prev_pts, tracks.curr_pts, cv::Mat());

    const auto prev_norm = undistort_to_normalized(tracks.prev_pts, config_.cam.K, config_.cam.D);
    const auto curr_norm = undistort_to_normalized(tracks.curr_pts, config_.cam.K, config_.cam.D);

    cv::Mat essential_mask;
    const cv::Mat E = cv::findEssentialMat(
        prev_norm,
        curr_norm,
        1.0,
        cv::Point2d(0.0, 0.0),
        cv::RANSAC,
        0.999,
        1.5e-3,
        essential_mask);
    if (E.empty()) {
        return out;
    }

    cv::Mat R;
    cv::Mat t;
    cv::Mat pose_mask = essential_mask.clone();
    const int pose_inliers = cv::recoverPose(
        E,
        prev_norm,
        curr_norm,
        R,
        t,
        1.0,
        cv::Point2d(0.0, 0.0),
        pose_mask);

    out.pose_inliers = pose_inliers;
    out.inlier_mask = pose_mask;
    out.median_flow_px = median_flow(tracks.prev_pts, tracks.curr_pts, pose_mask);

    if (pose_inliers < 30 || out.median_flow_px < 1.0) {
        return out;
    }

    out.R_21 = R;
    out.t_21 = normalized_or_zero(mat_to_vec3d(t));
    out.ok = cv::norm(out.t_21) > 0.0;
    return out;
}

cv::Vec3d VisualInertialOdometry::relative_motion_dir_world(const RelativePoseEstimate& estimate,
                                                            const cv::Mat& R_cw_ref) const {
    if (!estimate.ok || estimate.R_21.empty()) {
        return cv::Vec3d(0.0, 0.0, 0.0);
    }

    // OpenCV recoverPose devuelve R,t que cumplen X2 = R*X1 + t.
    // La direccion del centro de camara actual respecto a la referencia,
    // expresada en la camara de referencia, es C_2^1 = -R^T t.
    const cv::Mat motion_ref = -estimate.R_21.t() * vec3_to_mat(estimate.t_21);
    const cv::Mat R_wc_ref = R_cw_ref.t();
    return normalized_or_zero(mat_to_vec3d(R_wc_ref * motion_ref));
}

bool VisualInertialOdometry::try_relocalize_from_keyframes(
    const VisionFrame& frame,
    RelativePoseEstimate& estimate,
    cv::Vec3d& pos_candidate,
    cv::Mat* R_cw_candidate) const {
    if (R_cw_candidate == nullptr) {
        return false;
    }

    for (auto it = keyframes_.rbegin(); it != keyframes_.rend(); ++it) {
        estimate = estimate_relative_pose(it->gray, frame.gray);
        if (!estimate.ok) {
            continue;
        }

        const cv::Vec3d world_dir = relative_motion_dir_world(estimate, it->R_cw_ref);
        if (cv::norm(world_dir) <= 1e-9) {
            continue;
        }

        pos_candidate = it->visual_pos + world_dir;
        *R_cw_candidate = estimate.R_21 * it->R_cw_ref;
        return true;
    }
    return false;
}

bool VisualInertialOdometry::should_add_keyframe(const VisionFrame& frame,
                                                 const cv::Vec3d& pos_candidate) const {
    if (keyframes_.empty()) {
        return true;
    }

    const Keyframe& last = keyframes_.back();
    const double dist = cv::norm(pos_candidate - last.visual_pos);
    const double dt = frame.timestamp_sec - last.timestamp_sec;
    const std::uint64_t frame_gap = frame.frame_number - last.frame_number;

    return dist > 5.0 || dt > 0.75 || frame_gap >= 8;
}

void VisualInertialOdometry::maybe_add_keyframe(const VisionFrame& frame,
                                                const cv::Vec3d& pos_candidate,
                                                const cv::Mat& R_cw_ref) {
    if (!should_add_keyframe(frame, pos_candidate)) {
        return;
    }

    Keyframe kf;
    kf.gray = frame.gray.clone();
    kf.timestamp_sec = frame.timestamp_sec;
    kf.frame_number = frame.frame_number;
    kf.visual_pos = pos_candidate;
    kf.R_cw_ref = R_cw_ref.clone();
    keyframes_.push_back(kf);

    while (keyframes_.size() > 12) {
        keyframes_.pop_front();
    }
}

void VisualInertialOdometry::reset_scale_tracking(const cv::Vec3d& visual_pos,
                                                  const ImuCameraPose& imu_pose) {
    scale_anchor_visual_pos_ = visual_pos;
    scale_anchor_imu_pos_ = imu_pose.p_rel;
    scale_anchor_timestamp_sec_ = imu_pose.valid ? imu_pose.timestamp_sec : 0.0;
    scale_anchor_imu_timestamp_sec_ = imu_pose.valid ? imu_pose.timestamp_sec : 0.0;
    scale_anchor_initialized_ = imu_pose.valid;
}

bool VisualInertialOdometry::maybe_update_scale_estimate(double timestamp_sec,
                                                         double frame_dt,
                                                         const cv::Vec3d& visual_pos_current,
                                                         const ImuCameraPose& imu_cam_pose) {
    scale_updated_this_frame_ = false;
    if (!imu_cam_pose.valid) {
        return false;
    }
    if (imu_cam_pose.interval_dt <= 0.0) {
        return false;
    }

    const double dt_err = std::abs(imu_cam_pose.interval_dt - frame_dt);
    const double dt_tol = 0.02 + 0.35 * std::max(0.0, frame_dt);
    if (frame_dt > 0.0 && dt_err > dt_tol) {
        return false;
    }

    const double imu_age = std::abs(timestamp_sec - imu_cam_pose.timestamp_sec);
    const double imu_age_tol = 0.005 + 0.50 * std::max(0.0, frame_dt);
    if (imu_age > imu_age_tol) {
        return false;
    }

    if (!scale_anchor_initialized_) {
        reset_scale_tracking(visual_pos_current, imu_cam_pose);
        return false;
    }

    const double visual_step = cv::norm(visual_pos_current - scale_anchor_visual_pos_);
    const double imu_step = cv::norm(imu_cam_pose.p_rel - scale_anchor_imu_pos_);

    reset_scale_tracking(visual_pos_current, imu_cam_pose);

    if (visual_step < 0.25 || imu_step < 0.01) {
        return false;
    }

    scale_window_.push_back(ScaleWindowEntry{timestamp_sec, visual_step, imu_step});
    while (static_cast<int>(scale_window_.size()) > 20) {
        scale_window_.pop_front();
    }
    if (static_cast<int>(scale_window_.size()) < 4) {
        return false;
    }

    std::vector<double> ratios;
    ratios.reserve(scale_window_.size());
    for (const auto& entry : scale_window_) {
        if (entry.visual_step <= 1e-9 || entry.imu_step <= 1e-9) {
            continue;
        }
        ratios.push_back(entry.imu_step / entry.visual_step);
    }
    if (ratios.size() < 4) {
        return false;
    }

    const double median_ratio = robust_median(ratios);
    if (!std::isfinite(median_ratio) || median_ratio <= 1e-6) {
        return false;
    }

    std::vector<double> deviations;
    deviations.reserve(ratios.size());
    for (double ratio : ratios) {
        deviations.push_back(std::abs(ratio - median_ratio));
    }
    const double mad = robust_median(deviations);
    const double gate = std::max(0.20 * median_ratio, 2.5 * mad);

    double sum_visual = 0.0;
    double sum_imu = 0.0;
    int kept = 0;
    for (const auto& entry : scale_window_) {
        if (entry.visual_step <= 1e-9 || entry.imu_step <= 1e-9) {
            continue;
        }

        const double ratio = entry.imu_step / entry.visual_step;
        if (std::abs(ratio - median_ratio) > gate) {
            continue;
        }

        sum_visual += entry.visual_step;
        sum_imu += entry.imu_step;
        ++kept;
    }

    if (kept < 3 || sum_visual <= 1e-9 || sum_imu <= 1e-9) {
        return false;
    }

    const double candidate_scale = sum_imu / sum_visual;
    if (!std::isfinite(candidate_scale) || candidate_scale <= 1e-6) {
        return false;
    }

    if (!global_scale_initialized_) {
        global_scale_estimate_ = candidate_scale;
        global_scale_initialized_ = true;
    } else {
        const double alpha = clamp01(0.10 + 0.03 * static_cast<double>(kept));
        global_scale_estimate_ = (1.0 - alpha) * global_scale_estimate_ + alpha * candidate_scale;
    }
    scale_updated_this_frame_ = true;
    return true;
}

bool VisualInertialOdometry::current_imu_camera_pose(const ImuFrontendOutput* imu_state,
                                                     ImuCameraPose* pose) const {
    if (pose == nullptr) {
        return false;
    }
    *pose = ImuCameraPose{};

    if (imu_state == nullptr || !imu_state->initialized) {
        return false;
    }

    pose->timestamp_sec = imu_state->timestamp_sec;
    const cv::Mat R_wi = quat_wxyz_to_R(imu_state->quat_wxyz);
    cv::Mat R_wc = R_wi.clone();
    cv::Vec3d p_wc(imu_state->pos_xyz[0], imu_state->pos_xyz[1], imu_state->pos_xyz[2]);

    if (extrinsics_.valid) {
        R_wc = R_wi * extrinsics_.R_ic;
        p_wc += mat_to_vec3d(R_wi * vec3_to_mat(extrinsics_.t_ic));
    }

    pose->valid = true;
    pose->R_wc = R_wc;
    pose->R_cw = R_wc.t();
    pose->p_w = p_wc;
    pose->p_rel = p_wc;
    pose->rpy_deg = rotation_matrix_to_rpy_deg(R_wc);
    pose->interval_dt = imu_state->preint_dt;
    pose->preint_dtheta = imu_state->preint_dtheta;
    pose->preint_dv = imu_state->preint_dv;
    pose->preint_dp = imu_state->preint_dp;
    return true;
}

VioPoseOutput VisualInertialOdometry::make_output(const VisionFrame& frame,
                                                  bool valid_update,
                                                  const ImuCameraPose& imu_pose,
                                                  const cv::Mat& R_cw_out,
                                                  const cv::Mat& R_cw_visual,
                                                  const RelativePoseEstimate& estimate) const {
    VioPoseOutput out;
    out.initialized = initialized_;
    out.valid = valid_update;
    out.imu_used = imu_pose.valid;
    out.scale_initialized = global_scale_initialized_;
    out.scale_updated = scale_updated_this_frame_;
    out.frame_idx = static_cast<int>(frame.frame_number);
    out.timestamp_sec = (first_ts_ < 0.0) ? 0.0 : (frame.timestamp_sec - first_ts_);
    out.frame_dt = (prev_frame_ts_ < 0.0) ? 0.0 : (frame.timestamp_sec - prev_frame_ts_);
    out.fps = fps_ema_;

    const cv::Vec3d pos_metric = visual_position_unscaled_ * (global_scale_initialized_ ? global_scale_estimate_ : 1.0);
    out.x = pos_metric[0];
    out.y = pos_metric[1];
    out.z = pos_metric[2];

    out.x_visual_unscaled = visual_position_unscaled_[0];
    out.y_visual_unscaled = visual_position_unscaled_[1];
    out.z_visual_unscaled = visual_position_unscaled_[2];

    out.x_imu = imu_pose.p_rel[0];
    out.y_imu = imu_pose.p_rel[1];
    out.z_imu = imu_pose.p_rel[2];

    const cv::Vec3d rpy_out = rotation_matrix_to_rpy_deg(R_cw_out.t());
    const cv::Vec3d rpy_visual = rotation_matrix_to_rpy_deg(R_cw_visual.t());
    const cv::Vec3d rpy_imu = imu_pose.valid ? imu_pose.rpy_deg : rpy_visual;

    out.roll_deg = rpy_out[0];
    out.pitch_deg = rpy_out[1];
    out.yaw_deg = rpy_out[2];
    out.roll_deg_visual = rpy_visual[0];
    out.pitch_deg_visual = rpy_visual[1];
    out.yaw_deg_visual = rpy_visual[2];
    out.roll_deg_imu = rpy_imu[0];
    out.pitch_deg_imu = rpy_imu[1];
    out.yaw_deg_imu = rpy_imu[2];
    out.quat_wxyz = rotation_matrix_to_quat_wxyz(R_cw_out.t());

    out.raw_matches = estimate.tracks.raw_matches;
    out.tracked_points = estimate.tracks.tracked_points;
    out.pose_inliers = estimate.pose_inliers;
    out.median_flow_px = estimate.median_flow_px;
    out.scale_estimate = global_scale_estimate_;
    return out;
}

VioPoseOutput VisualInertialOdometry::update(const VisionFrame& frame,
                                             const ImuFrontendOutput* imu_state,
                                             cv::Mat* debug_vis) {
    update_fps();

    if (frame.gray.empty()) {
        throw std::runtime_error("VisionFrame invalido: gray vacio.");
    }
    if (frame.gray.cols != config_.cam.w || frame.gray.rows != config_.cam.h) {
        std::ostringstream oss;
        oss << "Tamano de frame inesperado. Esperado "
            << config_.cam.w << "x" << config_.cam.h
            << ", recibido " << frame.gray.cols << "x" << frame.gray.rows;
        throw std::runtime_error(oss.str());
    }

    if (first_ts_ < 0.0) {
        first_ts_ = frame.timestamp_sec;
    }
    scale_updated_this_frame_ = false;

    ImuCameraPose imu_pose;
    const bool imu_pose_valid = current_imu_camera_pose(imu_state, &imu_pose);
    if (imu_pose_valid) {
        if (!imu_cam_origin_initialized_) {
            imu_cam_origin_ = imu_pose.p_w;
            imu_cam_origin_initialized_ = true;
        }
        imu_pose.p_rel -= imu_cam_origin_;
    }

    RelativePoseEstimate estimate;
    cv::Mat R_cw_out = imu_pose_valid ? imu_pose.R_cw.clone() : R_cw_visual_.clone();

    if (!initialized_) {
        initialized_ = true;
        if (imu_pose_valid) {
            // Anchor the visual world to the initial IMU/camera attitude so
            // visual translation and inertial rotation share the same world frame.
            R_cw_visual_ = imu_pose.R_cw.clone();
            R_cw_out = imu_pose.R_cw.clone();
            reset_scale_tracking(visual_position_unscaled_, imu_pose);
        }
        prev_frame_ = frame;
        prev_frame_ts_ = frame.timestamp_sec;
        maybe_add_keyframe(frame, visual_position_unscaled_, R_cw_visual_);

        last_output_ = make_output(frame,
                                   false,
                                   imu_pose,
                                   R_cw_out,
                                   R_cw_visual_,
                                   estimate);
        if (debug_vis != nullptr) {
            draw_tracks(frame.bgr, {}, {}, cv::Mat(), last_output_, *debug_vis);
        }
        return last_output_;
    }

    const double frame_dt = (prev_frame_ts_ < 0.0) ? 0.0 : (frame.timestamp_sec - prev_frame_ts_);
    const cv::Mat R_cw_ref_prev = R_cw_visual_.clone();
    estimate = estimate_relative_pose(prev_frame_.gray, frame.gray);

    cv::Vec3d visual_pos_candidate = visual_position_unscaled_;
    cv::Mat R_cw_visual_candidate = R_cw_visual_.clone();
    bool relocalized = false;

    if (estimate.ok) {
        const cv::Vec3d world_dir = relative_motion_dir_world(estimate, R_cw_ref_prev);
        if (cv::norm(world_dir) > 1e-9) {
            visual_pos_candidate = visual_position_unscaled_ + world_dir;
            R_cw_visual_candidate = estimate.R_21 * R_cw_visual_;
        } else {
            estimate.ok = false;
        }
    }

    if (!estimate.ok) {
        relocalized = try_relocalize_from_keyframes(frame,
                                                    estimate,
                                                    visual_pos_candidate,
                                                    &R_cw_visual_candidate);
    }

    const bool valid_update = estimate.ok || relocalized;
    if (valid_update) {
        visual_position_unscaled_ = visual_pos_candidate;
        R_cw_visual_ = R_cw_visual_candidate;
        if (relocalized) {
            scale_window_.clear();
            if (imu_pose_valid) {
                reset_scale_tracking(visual_position_unscaled_, imu_pose);
            } else {
                scale_anchor_initialized_ = false;
            }
        } else if (estimate.ok) {
            maybe_update_scale_estimate(frame.timestamp_sec, frame_dt, visual_position_unscaled_, imu_pose);
        }
        maybe_add_keyframe(frame, visual_position_unscaled_, R_cw_visual_);
    }

    R_cw_out = imu_pose_valid ? imu_pose.R_cw.clone() : R_cw_visual_.clone();

    const cv::Vec3d pos_metric = visual_position_unscaled_ * (global_scale_initialized_ ? global_scale_estimate_ : 1.0);
    const cv::Mat t_out = -R_cw_out * vec3_to_mat(pos_metric);
    T_cw_ = make_pose(R_cw_out, t_out);

    last_output_ = make_output(frame,
                               valid_update,
                               imu_pose,
                               R_cw_out,
                               R_cw_visual_,
                               estimate);

    if (debug_vis != nullptr) {
        draw_tracks(frame.bgr,
                    estimate.tracks.prev_pts,
                    estimate.tracks.curr_pts,
                    estimate.inlier_mask,
                    last_output_,
                    *debug_vis);
    }

    prev_frame_ = frame;
    prev_frame_ts_ = frame.timestamp_sec;
    return last_output_;
}

} // namespace vo
