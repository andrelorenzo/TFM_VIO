#include "gt_est.hpp"
#include "lie_math.hpp"
#include <algorithm>

static bool g_init = false;
static Camera gcam;
static DepOrb gdorb;
static SourceIn last_frame;
static cv::Mat T_cw_;
static double g_last_frame_tsms = -1.0;

static cv::Mat toGray(cv::Mat color);
static bool isInFrame(const cv::Point2f& p, const cv::Mat & frame);
static double depthAt(const cv::Mat& depth, const cv::Point2f& p, int r);
static vec3 rotationToCameraEulerRad(const cv::Matx33d& R);


void gtInit(Config * config){
    if(!config->gen.groundt)return;

    if(!config->gen.color_on || !config->gen.groundt){
        Logger(ERROR, "[GT] Color frame is needed");
        return;
    }

    if(config->cam.K.empty()){
        Logger(ERROR, "[GT] Camera intrinsics are empty");
        return;
    }

    gcam  = config->cam;
    gdorb = config->dorb;
    T_cw_ = cv::Mat::eye(4, 4, CV_64F);
    g_last_frame_tsms = -1.0;

    g_init = true;
}


void gtUpdate(SourceIn * source, StateOut * state){
    if(!g_init)return;
    if(source->frame.empty() || source->depth.empty())return;
    if(source->frame_tsms <= 0.0)return;
    if(g_last_frame_tsms >= 0.0 && source->frame_tsms <= g_last_frame_tsms + 1e-9)return;

    g_last_frame_tsms = source->frame_tsms;

    cv::Mat curr_gray = toGray(source->frame);
    if(curr_gray.empty())return;

    if(last_frame.depth.empty() || last_frame.frame.empty()){
        Logger(WARN, "GT primed with first frame ts=%.3f", source->frame_tsms);
        last_frame = *source;
        last_frame.frame = curr_gray.clone();
        T_cw_ = cv::Mat::eye(4, 4, CV_64F);
        return;
    }

    std::vector<cv::Point2f> prev_pts;
    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    std::vector<float> err;
    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;

    cv::goodFeaturesToTrack(last_frame.frame, prev_pts, gdorb.max_feat, gdorb.quality, gdorb.min_dist);

    if((int)prev_pts.size() < gdorb.min_pts){
        last_frame = *source;
        last_frame.frame = curr_gray.clone();
        return;
    }

    // Optical flow on color frame
    cv::calcOpticalFlowPyrLK(last_frame.frame,curr_gray,prev_pts,curr_pts,status,err,cv::Size(gdorb.lk_win_size, gdorb.lk_win_size),gdorb.lk_max_lvl,cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));

    // Depth is already aligned to color, so 3D is reconstructed in color-camera coordinates.
    const double fx = gcam.K.at<double>(0, 0);
    const double fy = gcam.K.at<double>(1, 1);
    const double cx = gcam.K.at<double>(0, 2);
    const double cy = gcam.K.at<double>(1, 2);

    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (!status[i]) continue;
        if (!isInFrame(curr_pts[i], curr_gray)) continue;
        if (!isInFrame(prev_pts[i], last_frame.depth)) continue;

        const double u = prev_pts[i].x;
        const double v = prev_pts[i].y;
        const double z_d = depthAt(last_frame.depth, prev_pts[i], gdorb.patch_r);
        if (!std::isfinite(z_d)) continue;

        const double x = (u - cx) / fx * z_d;
        const double y = (v - cy) / fy * z_d;
        if (!std::isfinite(x) || !std::isfinite(y) || z_d <= 0.0) continue;

        points3d.emplace_back((float)x, (float)y, (float)z_d);
        points2d.push_back(curr_pts[i]);
    }


    if((int)points3d.size() < gdorb.min_pts){
        last_frame = *source;
        last_frame.frame = curr_gray.clone();
        return;
    }

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;

    // Get camera pose
    const bool ok = cv::solvePnPRansac(points3d,points2d,gcam.K,gcam.D,rvec,tvec,false,200,3.0,0.999,inliers,cv::SOLVEPNP_ITERATIVE);

    const int pinliers = inliers.empty() ? 0 : inliers.rows;
    if (!ok || pinliers < gdorb.min_inliers) {
        last_frame = *source;
        last_frame.frame = curr_gray.clone();
        return;
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    cv::Mat T_rel = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T_rel(cv::Range(0,3), cv::Range(0,3)));
    tvec.copyTo(T_rel(cv::Range(0,3), cv::Range(3,4)));

    T_cw_ = T_rel * T_cw_;

    const cv::Mat R_cw = T_cw_(cv::Range(0, 3), cv::Range(0, 3));
    const cv::Mat t_cw = T_cw_(cv::Range(0, 3), cv::Range(3, 4));
    const cv::Mat R_wc = R_cw.t();
    const cv::Mat p_w = -R_wc * t_cw;

    cv::Mat R_wc_d;
    R_wc.convertTo(R_wc_d, CV_64F);

    cv::Matx33d R_wc_x;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_wc_x(r, c) = R_wc.at<double>(r, c);
        }
    }

    state->posgt_m = vec3( p_w.at<double>(0, 0), p_w.at<double>(1, 0), p_w.at<double>(2, 0));
    state->origt_rad = rotationToCameraEulerRad(R_wc_x);

    last_frame = *source;
    last_frame.frame = curr_gray.clone();
}

static vec3 rotationToCameraEulerRad(const cv::Matx33d& R){
    mat3 Re;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            Re(r, c) = R(r, c);
        }
    }
    return quatToCameraRpyRad(normalizeQ(quat(Re)));
}

static cv::Mat toGray(cv::Mat color){
    if(color.empty())return cv::Mat();
    if(color.channels() == 1)return color.clone();

    cv::Mat out;
    cv::cvtColor(color, out, cv::COLOR_BGR2GRAY);

    return out.clone();
}


static bool isInFrame(const cv::Point2f& p, const cv::Mat & frame){
    return p.x >= 0.0f && p.y >= 0.0f && p.x < frame.size().width && p.y < frame.size().height;
}

static double depthAt(const cv::Mat& depth, const cv::Point2f& p, int r){
    if (depth.empty() || depth.type() != CV_32F) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if(std::lround(p.x) < 0 || std::lround(p.y) < 0 || std::lround(p.x) > depth.cols || std::lround(p.y) > depth.rows){
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<float> values;
    values.reserve((2 * r + 1) * (2 * r + 1));

    int x0 = std::max((int)0, (int)(std::lround(p.x) - r));
    int x1 = std::min((int)(depth.cols - 1), (int)(std::lround(p.x) + r));
    int y0 = std::max((int)0, (int)(std::lround(p.y) - r));
    int y1 = std::min((int)(depth.rows - 1), (int)(std::lround(p.y) + r));

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float d = depth.at<float>(y, x);
            if (std::isfinite(d) && d > 0.05f && d < 20.0f) {
                values.push_back(d);
            }
        }
    }

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
