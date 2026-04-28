#include "gt_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool GroundTruthEstimator::init(const GroundTruthConfig& cfg)
{
    close();
    cfg_ = cfg;
    T_cw_ = identityPose();
    initialized_ = true;
    return true;
}

void GroundTruthEstimator::reset()
{
    has_prev_frame_ = false;
    prev_frame_ = GroundTruthFrame{};
    T_cw_ = identityPose();
}

void GroundTruthEstimator::close()
{
    initialized_ = false;
    has_prev_frame_ = false;
    prev_frame_ = GroundTruthFrame{};
    T_cw_.release();
}

double GroundTruthEstimator::computeMedian(std::vector<float>& values)
{
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const size_t n = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + n, values.end());
    double med = static_cast<double>(values[n]);

    if ((values.size() % 2) == 0U) {
        std::nth_element(values.begin(), values.begin() + n - 1, values.end());
        med = 0.5 * (med + static_cast<double>(values[n - 1]));
    }

    return med;
}

cv::Mat GroundTruthEstimator::identityPose()
{
    return cv::Mat::eye(4, 4, CV_64F);
}

cv::Mat GroundTruthEstimator::makePose(const cv::Mat& R, const cv::Mat& t)
{
    cv::Mat T = identityPose();
    R.copyTo(T(cv::Range(0, 3), cv::Range(0, 3)));
    t.copyTo(T(cv::Range(0, 3), cv::Range(3, 4)));
    return T;
}

bool GroundTruthEstimator::insideImage(const cv::Point2f& p, const cv::Size& size)
{
    return p.x >= 0.0f && p.y >= 0.0f &&
           p.x < static_cast<float>(size.width) &&
           p.y < static_cast<float>(size.height);
}

double GroundTruthEstimator::sampleDepthPatch(const cv::Mat& depth_m,
                                              const cv::Point2f& px,
                                              int radius)
{
    if (depth_m.empty() || depth_m.type() != CV_32F) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const int cx = static_cast<int>(std::lround(px.x));
    const int cy = static_cast<int>(std::lround(px.y));
    const int width = depth_m.cols;
    const int height = depth_m.rows;

    if (cx < 0 || cy < 0 || cx >= width || cy >= height) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::vector<float> values;
    values.reserve((2 * radius + 1) * (2 * radius + 1));

    const int x0 = std::max(0, cx - radius);
    const int x1 = std::min(width - 1, cx + radius);
    const int y0 = std::max(0, cy - radius);
    const int y1 = std::min(height - 1, cy + radius);

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float d = depth_m.at<float>(y, x);
            if (std::isfinite(d) && d > 0.05f && d < 20.0f) {
                values.push_back(d);
            }
        }
    }

    return computeMedian(values);
}

cv::Matx44d GroundTruthEstimator::matToMatx44d(const cv::Mat& T)
{
    cv::Mat Td;
    T.convertTo(Td, CV_64F);

    cv::Matx44d out = cv::Matx44d::eye();
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out(r, c) = Td.at<double>(r, c);
        }
    }
    return out;
}

cv::Vec4d GroundTruthEstimator::rotationToQuaternion(const cv::Matx33d& R)
{
    const double trace = R(0,0) + R(1,1) + R(2,2);

    double qw = 1.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;

    if (trace > 0.0) {
        const double s = 0.5 / std::sqrt(trace + 1.0);
        qw = 0.25 / s;
        qx = (R(2,1) - R(1,2)) * s;
        qy = (R(0,2) - R(2,0)) * s;
        qz = (R(1,0) - R(0,1)) * s;
    } else if (R(0,0) > R(1,1) && R(0,0) > R(2,2)) {
        const double s = 2.0 * std::sqrt(1.0 + R(0,0) - R(1,1) - R(2,2));
        qw = (R(2,1) - R(1,2)) / s;
        qx = 0.25 * s;
        qy = (R(0,1) + R(1,0)) / s;
        qz = (R(0,2) + R(2,0)) / s;
    } else if (R(1,1) > R(2,2)) {
        const double s = 2.0 * std::sqrt(1.0 + R(1,1) - R(0,0) - R(2,2));
        qw = (R(0,2) - R(2,0)) / s;
        qx = (R(0,1) + R(1,0)) / s;
        qy = 0.25 * s;
        qz = (R(1,2) + R(2,1)) / s;
    } else {
        const double s = 2.0 * std::sqrt(1.0 + R(2,2) - R(0,0) - R(1,1));
        qw = (R(1,0) - R(0,1)) / s;
        qx = (R(0,2) + R(2,0)) / s;
        qy = (R(1,2) + R(2,1)) / s;
        qz = 0.25 * s;
    }

    const double n = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    if (n < 1e-12) {
        return cv::Vec4d(1.0, 0.0, 0.0, 0.0);
    }

    return cv::Vec4d(qw / n, qx / n, qy / n, qz / n);
}

cv::Vec3d GroundTruthEstimator::rotationToEulerRad(const cv::Matx33d& R)
{
    const double sy = std::sqrt(R(0,0) * R(0,0) + R(1,0) * R(1,0));
    const bool singular = sy < 1e-9;

    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;

    if (!singular) {
        roll = std::atan2(R(2,1), R(2,2));
        pitch = std::atan2(-R(2,0), sy);
        yaw = std::atan2(R(1,0), R(0,0));
    } else {
        roll = std::atan2(-R(1,2), R(1,1));
        pitch = std::atan2(-R(2,0), sy);
        yaw = 0.0;
    }

    return cv::Vec3d(roll, pitch, yaw);
}

GroundTruthState GroundTruthEstimator::update(const GroundTruthFrame& frame)
{
    if (!initialized_) {
        throw std::runtime_error("GroundTruthEstimator::init() must be called first");
    }

    GroundTruthState out;
    out.timestamp_sec = frame.timestamp_sec;
    out.frame_number = frame.frame_number;

    if (!cfg_.enabled) {
        return out;
    }

    if (frame.gray.empty() || frame.depth_m.empty() || frame.K.empty()) {
        return out;
    }

    if (!has_prev_frame_) {
        prev_frame_ = frame;
        has_prev_frame_ = true;
        out.initialized = true;
        return out;
    }

    std::vector<cv::Point2f> prev_pts;
    cv::goodFeaturesToTrack(prev_frame_.gray,
                            prev_pts,
                            cfg_.max_features,
                            0.01,
                            8.0);

    if (static_cast<int>(prev_pts.size()) < cfg_.min_points) {
        prev_frame_ = frame;
        return out;
    }

    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(prev_frame_.gray,
                             frame.gray,
                             prev_pts,
                             curr_pts,
                             status,
                             err,
                             cv::Size(cfg_.lk_win_size, cfg_.lk_win_size),
                             cfg_.lk_max_level,
                             cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));

    const double fx = frame.K.at<double>(0, 0);
    const double fy = frame.K.at<double>(1, 1);
    const double cx = frame.K.at<double>(0, 2);
    const double cy = frame.K.at<double>(1, 2);

    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;
    points3d.reserve(prev_pts.size());
    points2d.reserve(prev_pts.size());

    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (!status[i] || !insideImage(curr_pts[i], frame.gray.size())) {
            continue;
        }

        const double depth_m = sampleDepthPatch(prev_frame_.depth_m, prev_pts[i], cfg_.depth_patch_radius);
        if (!std::isfinite(depth_m)) {
            continue;
        }

        const double x = (prev_pts[i].x - cx) / fx * depth_m;
        const double y = (prev_pts[i].y - cy) / fy * depth_m;
        const double z = depth_m;

        points3d.emplace_back(static_cast<float>(x),
                              static_cast<float>(y),
                              static_cast<float>(z));
        points2d.push_back(curr_pts[i]);
    }

    out.tracked_points = static_cast<int>(points3d.size());
    if (static_cast<int>(points3d.size()) < cfg_.min_points) {
        prev_frame_ = frame;
        return out;
    }

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;
    const bool ok = cv::solvePnPRansac(points3d,points2d,frame.K,frame.dist,rvec,tvec,false,200,3.0,0.999,inliers,cv::SOLVEPNP_ITERATIVE);

    out.pnp_inliers = inliers.empty() ? 0 : inliers.rows;
    if (!ok || out.pnp_inliers < cfg_.min_inliers) {
        prev_frame_ = frame;
        return out;
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    const cv::Mat T_rel = makePose(R, tvec);
    T_cw_ = T_rel * T_cw_;

    const cv::Mat R_cw = T_cw_(cv::Range(0, 3), cv::Range(0, 3));
    const cv::Mat t_cw = T_cw_(cv::Range(0, 3), cv::Range(3, 4));
    const cv::Mat R_wc = R_cw.t();
    const cv::Mat p_w = -R_wc * t_cw;

    cv::Mat T_wc_mat = identityPose();
    R_wc.copyTo(T_wc_mat(cv::Range(0, 3), cv::Range(0, 3)));
    p_w.copyTo(T_wc_mat(cv::Range(0, 3), cv::Range(3, 4)));

    cv::Mat R_wc_d;
    R_wc.convertTo(R_wc_d, CV_64F);
    cv::Matx33d R_wc_x;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_wc_x(r, c) = R_wc_d.at<double>(r, c);
        }
    }

    out.valid = true;
    out.xyz = cv::Vec3d(p_w.at<double>(0, 0),
                        p_w.at<double>(1, 0),
                        p_w.at<double>(2, 0));
    out.rpy_rad = rotationToEulerRad(R_wc_x);
    out.rpy_deg = out.rpy_rad * (180.0 / M_PI);
    out.quat = rotationToQuaternion(R_wc_x);
    out.T_wc = matToMatx44d(T_wc_mat);

    prev_frame_ = frame;
    return out;
}
