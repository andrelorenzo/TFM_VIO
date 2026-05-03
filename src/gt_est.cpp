#include "gt_est.hpp"
#include "lie_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

// Cached camera intrinsics / distortion copied from Config.
static Camera cam_config;

// Previous packet used as reference for the next RGB-D relative estimate.
static SourceIn last_frame;

// Accumulated transform: world -> current camera.
static Eigen::Isometry3d T_cw = Eigen::Isometry3d::Identity();

static bool gt_inited = false;
struct DepthParams {
    int max_feat = 2000;
    int min_pts = 80;
    int min_inliers = 40;
    int lk_max_lvl = 6;
    int lk_win_size = 21;
    int patch_r = 1;

    double min_dist = 8.0;
    double quality = 0.01;
};

static DepthParams dorb;

static cv::Mat toGray(const cv::Mat& color);
static bool isInFrame(const cv::Point2f& p, const cv::Mat& frame);
static double depthAt(const cv::Mat& depth, const cv::Point2f& p, int r);
static void storeReferenceFrame(const SourceIn& source, const cv::Mat& gray);
static bool backprojectPrevPoint( const cv::Point2f& px, double depth, cv::Point3f& out);


// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

void gtInit(Config* config)
{
    if (!config->gen.color_on || !config->gen.depth_on) {
        Logger(WARN, "[GT] Color and depth frames are needed");
        return;
    }

    if (config->cam.K.empty()) {
        Logger(ERROR, "[GT] Camera intrinsics are empty");
        return;
    }

    cam_config = config->cam;
    T_cw = Eigen::Isometry3d::Identity();

    last_frame = SourceIn();

    gt_inited = true;
}


void gtUpdate(SourceIn* source, StateOut* state)
{
    if(!gt_inited)return;
    
    if ( !source || !state) {
        return;
    }

    if (source->frame.empty() || source->depth.empty()) {
        return;
    }

    const cv::Mat gray = toGray(source->frame);
    if (gray.empty()) {
        return;
    }

    // First valid frame only initializes the world origin.
    if (last_frame.frame.empty() || last_frame.depth.empty()) {
        Logger(WARN, "GT primed with first frame ts=%.3f", source->frame_tsms);

        T_cw = Eigen::Isometry3d::Identity();
        storeReferenceFrame(*source, gray);
        return;
    }

    std::vector<cv::Point2f> prev_pts;
    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::goodFeaturesToTrack(last_frame.frame,prev_pts,dorb.max_feat,dorb.quality,dorb.min_dist);

    if (static_cast<int>(prev_pts.size()) < dorb.min_pts) {
        storeReferenceFrame(*source, gray);
        return;
    }

    cv::calcOpticalFlowPyrLK( last_frame.frame, gray, prev_pts, curr_pts, status, err, cv::Size(dorb.lk_win_size, dorb.lk_win_size), dorb.lk_max_lvl, cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,30,0.01));

    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;

    points3d.reserve(prev_pts.size());
    points2d.reserve(prev_pts.size());

    for (size_t i = 0; i < prev_pts.size(); ++i) {
        if (!status[i] || !isInFrame(prev_pts[i], last_frame.depth) || !isInFrame(curr_pts[i], gray)) {
            continue;
        }

        cv::Point3f X_prev;
        const double z = depthAt(last_frame.depth, prev_pts[i], dorb.patch_r);

        if (!std::isfinite(z) || !backprojectPrevPoint(prev_pts[i], z, X_prev)) {
            continue;
        }

        points3d.push_back(X_prev);
        points2d.push_back(curr_pts[i]);
    }

    if (static_cast<int>(points3d.size()) < dorb.min_pts) {
        storeReferenceFrame(*source, gray);
        return;
    }

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;

    const bool ok = cv::solvePnPRansac(points3d,points2d,cam_config.K,cam_config.D,rvec,tvec,false,200,3.0,0.999,inliers,cv::SOLVEPNP_ITERATIVE);

    const int num_inliers = inliers.empty() ? 0 : inliers.rows;

    if (!ok || num_inliers < dorb.min_inliers) {
        storeReferenceFrame(*source, gray);
        return;
    }
    
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = cvMat33ToEigen(R_cv);
    T.translation() = cvVec3ToEigen(tvec);
    T_cw = T * T_cw;

    // Camera pose in world frame.
    const Eigen::Isometry3d T_wc = T_cw.inverse();

    const vec3 p_wc = T_wc.translation();
    const mat3 R_wc = T_wc.linear();

    state->gtpose.pos = p_wc;
    state->gtpose.rot = normalizeQ(quat(R_wc));
    storeReferenceFrame(*source, gray);
}




static cv::Mat toGray(const cv::Mat& color)
{
    if (color.empty()) {
        return cv::Mat();
    }

    if (color.channels() == 1) {
        return color.clone();
    }

    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    return gray;
}
static bool isInFrame(const cv::Point2f& p, const cv::Mat& frame) {
    return !frame.empty() && p.x >= 0.0f && p.y >= 0.0f && p.x < static_cast<float>(frame.cols) && p.y < static_cast<float>(frame.rows);
}
static double depthAt(const cv::Mat& depth, const cv::Point2f& p, int r)
{
    if (depth.empty() || depth.type() != CV_32F) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const int cx = static_cast<int>(std::lround(p.x));
    const int cy = static_cast<int>(std::lround(p.y));

    // Importante: aquí debe ser >=, no >.
    if (cx < 0 || cy < 0 || cx >= depth.cols || cy >= depth.rows) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const int x0 = std::max(0, cx - r);
    const int x1 = std::min(depth.cols - 1, cx + r);
    const int y0 = std::max(0, cy - r);
    const int y1 = std::min(depth.rows - 1, cy + r);

    std::vector<float> values;
    values.reserve(static_cast<size_t>((x1 - x0 + 1) * (y1 - y0 + 1)));

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

    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());

    double med = static_cast<double>(values[mid]);

    if ((values.size() % 2U) == 0U) {
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        med = 0.5 * (med + static_cast<double>(values[mid - 1]));
    }

    return med;
}
static void storeReferenceFrame(const SourceIn& source, const cv::Mat& gray)
{
    last_frame = source;

    // Guardamos gris como frame de referencia.
    last_frame.frame = gray.clone();

    // Mejor clonar profundidad para evitar problemas si SourceIn reutiliza buffers.
    last_frame.depth = source.depth.clone();
}
static bool backprojectPrevPoint( const cv::Point2f& px, double depth, cv::Point3f& out){
    if (!std::isfinite(depth) || depth <= 0.0)return false;
        
    

    const double x = (static_cast<double>(px.x) - cam_config.getCx()) / cam_config.getFx() * depth;
    const double y = (static_cast<double>(px.y) - cam_config.getCy()) / cam_config.getFy() * depth;
    const double z = depth;

    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;


    out = cv::Point3f( static_cast<float>(x), static_cast<float>(y), static_cast<float>(z) );

    return true;
}
