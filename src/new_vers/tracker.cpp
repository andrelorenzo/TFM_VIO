#include "tracker.hpp"
#include "csv_logger.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

static Config g_tracker_cfg;
static bool g_tracker_init = false;
static int g_tracker_next_id = 0;
static int g_tracker_image_id = -1;
static cv::Mat g_prev_gray;
static std::vector<int> g_prev_ids;
static std::vector<cv::Point2f> g_prev_px;
static std::unordered_map<int, int> g_track_len_by_id;

static cv::Mat toGray(const cv::Mat& frame_bgr) {
    if (frame_bgr.empty()) return cv::Mat();
    if (frame_bgr.channels() == 1) return frame_bgr.clone();
    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

static bool insideFrame(const cv::Point2f& p, const cv::Mat& im) {
    return p.x >= 0.0f && p.y >= 0.0f && p.x < im.cols && p.y < im.rows;
}

static void undistortPointsNorm(const std::vector<cv::Point2f>& src, std::vector<cv::Point2f>* dst) {
    dst->clear();
    if (src.empty()) return;
    cv::undistortPoints(src, *dst, g_tracker_cfg.cam.K, g_tracker_cfg.cam.D);
}

static void detectNewPoints(const cv::Mat& gray,
                            const std::vector<cv::Point2f>& occupied,
                            std::vector<cv::Point2f>* pts) {
    pts->clear();
    if (gray.empty()) return;

    const int max_feat = std::max(0, g_tracker_cfg.dorb.max_feat - static_cast<int>(occupied.size()));
    if (max_feat <= 0) return;

    cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(255));
    const int min_dist = std::max(3, static_cast<int>(std::round(g_tracker_cfg.dorb.min_dist)));
    for (const cv::Point2f& p : occupied) {
        cv::circle(mask, p, min_dist, cv::Scalar(0), -1);
    }

    cv::goodFeaturesToTrack(
        gray,
        *pts,
        max_feat,
        std::max(1e-4, static_cast<double>(g_tracker_cfg.dorb.quality)),
        std::max(3.0, static_cast<double>(g_tracker_cfg.dorb.min_dist)),
        mask,
        std::max(3, g_tracker_cfg.trk.gftt_block_size),
        false,
        g_tracker_cfg.trk.gftt_k);

    if (!pts->empty()) {
        cv::cornerSubPix(
            gray,
            *pts,
            cv::Size(std::max(3, g_tracker_cfg.dorb.lk_win_size), std::max(3, g_tracker_cfg.dorb.lk_win_size)),
            cv::Size(-1, -1),
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                             std::max(5, g_tracker_cfg.trk.lk_iters),
                             std::max(1e-6, g_tracker_cfg.trk.lk_eps)));
    }
}

void trackerInit(Config * config){
    if (config == nullptr) return;
    g_tracker_cfg = *config;
    trackerReset();
    g_tracker_init = true;
}

void trackerReset() {
    g_tracker_next_id = 0;
    g_tracker_image_id = -1;
    g_prev_gray.release();
    g_prev_ids.clear();
    g_prev_px.clear();
    g_track_len_by_id.clear();
}

bool trackerTrackFrame(const cv::Mat& frame_bgr, TrackerOutput* out){
    if (!g_tracker_init || out == nullptr) return false;

    *out = TrackerOutput{};
    cv::Mat gray = toGray(frame_bgr);
    if (gray.empty()) return false;

    ++g_tracker_image_id;
    out->image_id = g_tracker_image_id;

    if (g_prev_gray.empty() || g_prev_px.empty()) {
        std::vector<cv::Point2f> new_px;
        detectNewPoints(gray, {}, &new_px);
        std::vector<cv::Point2f> new_un;
        undistortPointsNorm(new_px, &new_un);

        out->first_frame = true;
        out->new_px = new_px;
        out->new_un = new_un;
        out->new_ids.reserve(new_px.size());
        out->new_track_len.reserve(new_px.size());

        g_prev_ids.clear();
        g_prev_px = new_px;
        for (size_t i = 0; i < new_px.size(); ++i) {
            const int id = g_tracker_next_id++;
            out->new_ids.push_back(id);
            out->new_track_len.push_back(1);
            g_prev_ids.push_back(id);
            g_track_len_by_id[id] = 1;
        }

        g_prev_gray = gray;
        Logger(DEBUG, "[TRK] img=%d first=1 prev=0 tracked=0 new=%zu lost=0", out->image_id, out->new_ids.size());
        return true;
    }

    std::vector<cv::Point2f> curr_px;
    std::vector<uchar> lk_status;
    std::vector<float> lk_err;
    cv::calcOpticalFlowPyrLK(
        g_prev_gray,
        gray,
        g_prev_px,
        curr_px,
        lk_status,
        lk_err,
        cv::Size(std::max(5, g_tracker_cfg.dorb.lk_win_size), std::max(5, g_tracker_cfg.dorb.lk_win_size)),
        std::max(0, g_tracker_cfg.dorb.lk_max_lvl),
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                         std::max(5, g_tracker_cfg.trk.lk_iters),
                         std::max(1e-6, g_tracker_cfg.trk.lk_eps)));

    std::vector<int> kept_ids;
    std::vector<cv::Point2f> kept_prev_px;
    std::vector<cv::Point2f> kept_curr_px;
    out->lost_ids.clear();
    out->lost_track_len.clear();

    const int n_prev_points = static_cast<int>(g_prev_px.size());

    for (size_t i = 0; i < g_prev_px.size(); ++i) {
        if (i >= curr_px.size() || i >= lk_status.size() || !lk_status[i] || !insideFrame(curr_px[i], gray)) {
            out->lost_ids.push_back(g_prev_ids[i]);
            const auto it_len = g_track_len_by_id.find(g_prev_ids[i]);
            out->lost_track_len.push_back(it_len != g_track_len_by_id.end() ? it_len->second : 0);
            g_track_len_by_id.erase(g_prev_ids[i]);
            continue;
        }
        kept_ids.push_back(g_prev_ids[i]);
        kept_prev_px.push_back(g_prev_px[i]);
        kept_curr_px.push_back(curr_px[i]);
    }

    if (!kept_curr_px.empty()) {
        std::vector<cv::Point2f> back_px;
        std::vector<uchar> back_status;
        std::vector<float> back_err;
        cv::calcOpticalFlowPyrLK(
            gray,
            g_prev_gray,
            kept_curr_px,
            back_px,
            back_status,
            back_err,
            cv::Size(std::max(5, g_tracker_cfg.dorb.lk_win_size), std::max(5, g_tracker_cfg.dorb.lk_win_size)),
            std::max(0, g_tracker_cfg.dorb.lk_max_lvl),
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                             std::max(5, g_tracker_cfg.trk.lk_iters),
                             std::max(1e-6, g_tracker_cfg.trk.lk_eps)));

        std::vector<int> fb_ids;
        std::vector<cv::Point2f> fb_prev_px;
        std::vector<cv::Point2f> fb_curr_px;
        const double fb_max_err_px = std::max(0.1, g_tracker_cfg.trk.fb_max_err_px);

        for (size_t i = 0; i < kept_curr_px.size(); ++i) {
            if (i >= back_px.size() || i >= back_status.size() || !back_status[i]) {
                out->lost_ids.push_back(kept_ids[i]);
                const auto it_len = g_track_len_by_id.find(kept_ids[i]);
                out->lost_track_len.push_back(it_len != g_track_len_by_id.end() ? it_len->second : 0);
                g_track_len_by_id.erase(kept_ids[i]);
                continue;
            }

            const double fb_err = cv::norm(back_px[i] - kept_prev_px[i]);
            if (!std::isfinite(fb_err) || fb_err > fb_max_err_px) {
                out->lost_ids.push_back(kept_ids[i]);
                const auto it_len = g_track_len_by_id.find(kept_ids[i]);
                out->lost_track_len.push_back(it_len != g_track_len_by_id.end() ? it_len->second : 0);
                g_track_len_by_id.erase(kept_ids[i]);
                continue;
            }

            fb_ids.push_back(kept_ids[i]);
            fb_prev_px.push_back(kept_prev_px[i]);
            fb_curr_px.push_back(kept_curr_px[i]);
        }

        kept_ids.swap(fb_ids);
        kept_prev_px.swap(fb_prev_px);
        kept_curr_px.swap(fb_curr_px);
    }

    int n_f_inliers = -1;
    if (kept_curr_px.size() >= static_cast<size_t>(std::max(5, g_tracker_cfg.trk.min_ransac_points))) {
        std::vector<uchar> ransac_mask;
        cv::findFundamentalMat(kept_prev_px, kept_curr_px, cv::FM_RANSAC, std::max(0.1, g_tracker_cfg.trk.ransac_f_px), std::clamp(g_tracker_cfg.trk.ransac_conf, 0.5, 0.9999), ransac_mask);
        if (ransac_mask.size() == kept_curr_px.size()) {
            n_f_inliers = cv::countNonZero(ransac_mask);
            if (n_f_inliers >= std::max(5, g_tracker_cfg.trk.min_ransac_points)) {
                std::vector<int> inlier_ids;
                std::vector<cv::Point2f> inlier_prev_px;
                std::vector<cv::Point2f> inlier_curr_px;
                for (size_t i = 0; i < kept_curr_px.size(); ++i) {
                    if (ransac_mask[i]) {
                        inlier_ids.push_back(kept_ids[i]);
                        inlier_prev_px.push_back(kept_prev_px[i]);
                        inlier_curr_px.push_back(kept_curr_px[i]);
                    }
                }
                kept_ids.swap(inlier_ids);
                kept_prev_px.swap(inlier_prev_px);
                kept_curr_px.swap(inlier_curr_px);
            }
        }
    }

    out->tracked_ids = kept_ids;
    out->tracked_track_len.clear();
    out->tracked_track_len.reserve(kept_ids.size());
    for (const int id : kept_ids) {
        int& len = g_track_len_by_id[id];
        len = std::max(1, len) + 1;
        out->tracked_track_len.push_back(len);
    }
    out->tracked_prev_px = kept_prev_px;
    out->tracked_px = kept_curr_px;
    undistortPointsNorm(out->tracked_prev_px, &out->tracked_prev_un);
    undistortPointsNorm(out->tracked_px, &out->tracked_un);

    std::vector<cv::Point2f> new_px;
    detectNewPoints(gray, kept_curr_px, &new_px);
    out->new_px = new_px;
    undistortPointsNorm(out->new_px, &out->new_un);
    out->new_ids.reserve(out->new_px.size());
    out->new_track_len.reserve(out->new_px.size());

    g_prev_ids = kept_ids;
    g_prev_px = kept_curr_px;

    for (size_t i = 0; i < out->new_px.size(); ++i) {
        const int id = g_tracker_next_id++;
        out->new_ids.push_back(id);
        out->new_track_len.push_back(1);
        g_prev_ids.push_back(id);
        g_prev_px.push_back(out->new_px[i]);
        g_track_len_by_id[id] = 1;
    }

    g_prev_gray = gray;
    Logger(DEBUG, "[TRK] img=%d first=0 prev=%d tracked=%zu new=%zu lost=%zu f_inl=%d", out->image_id, n_prev_points, out->tracked_ids.size(), out->new_ids.size(), out->lost_ids.size(), n_f_inliers);
    return true;
}
