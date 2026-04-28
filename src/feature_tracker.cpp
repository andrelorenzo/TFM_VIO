#include "feature_tracker.hpp"

#include <cmath>
#include <unordered_set>

#include <opencv2/video/tracking.hpp>

namespace vo {
namespace {

bool inside_image(const cv::Point2f& p, const cv::Size& size) {
    return p.x >= 0.0f && p.y >= 0.0f &&
           p.x < static_cast<float>(size.width) &&
           p.y < static_cast<float>(size.height);
}

} // namespace

OrbFlowTracker::OrbFlowTracker(const Config& config)
    : matcher_(cv::NORM_HAMMING, false) {
    nfeatures_ = (config.orb.nFeatures > 0) ? config.orb.nFeatures : 1200;
    const float scale_factor = (config.orb.scaleFactor > 1.0f) ? config.orb.scaleFactor : 1.2f;
    const int nlevels = (config.orb.nLevels > 0) ? config.orb.nLevels : 8;
    ini_th_fast_ = (config.orb.iniThFAST > 0) ? config.orb.iniThFAST : 20;
    min_th_fast_ = (config.orb.minThFAST > 0) ? config.orb.minThFAST : 7;

    orb_ = cv::ORB::create(
        nfeatures_,
        scale_factor,
        nlevels,
        31,
        0,
        2,
        cv::ORB::HARRIS_SCORE,
        31,
        ini_th_fast_);

    min_matches_ = std::max(30, std::min(nfeatures_ / 20, 100));
    ratio_test_ = 0.75;
    lk_win_size_ = 21;
    lk_max_level_ = 3;
    fb_max_error_px_ = 1.5;
    retry_min_keypoints_ = std::max(80, nfeatures_ / 10);
}

void OrbFlowTracker::detectAndDescribe(const cv::Mat& gray,
                                       std::vector<cv::KeyPoint>& kps,
                                       cv::Mat& desc,
                                       int fast_threshold) const {
    orb_->setFastThreshold(fast_threshold);
    orb_->detectAndCompute(gray, cv::noArray(), kps, desc);
}

bool OrbFlowTracker::track(const cv::Mat& prev_gray, const cv::Mat& curr_gray, TrackSet& out) const {
    out = TrackSet{};

    if (prev_gray.empty() || curr_gray.empty()) {
        return false;
    }

    std::vector<cv::KeyPoint> prev_kps;
    std::vector<cv::KeyPoint> curr_kps;
    cv::Mat prev_desc;
    cv::Mat curr_desc;

    detectAndDescribe(prev_gray, prev_kps, prev_desc, ini_th_fast_);
    if (static_cast<int>(prev_kps.size()) < retry_min_keypoints_ && min_th_fast_ < ini_th_fast_) {
        detectAndDescribe(prev_gray, prev_kps, prev_desc, min_th_fast_);
    }

    detectAndDescribe(curr_gray, curr_kps, curr_desc, ini_th_fast_);
    if (static_cast<int>(curr_kps.size()) < retry_min_keypoints_ && min_th_fast_ < ini_th_fast_) {
        detectAndDescribe(curr_gray, curr_kps, curr_desc, min_th_fast_);
    }

    orb_->setFastThreshold(ini_th_fast_);

    if (prev_desc.empty() || curr_desc.empty()) {
        return false;
    }

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_.knnMatch(prev_desc, curr_desc, knn_matches, 2);

    std::vector<cv::Point2f> prev_pts;
    std::vector<cv::Point2f> curr_guess;
    prev_pts.reserve(knn_matches.size());
    curr_guess.reserve(knn_matches.size());

    std::unordered_set<int> used_train;
    for (const auto& pair : knn_matches) {
        if (pair.size() < 2) {
            continue;
        }

        const cv::DMatch& m0 = pair[0];
        const cv::DMatch& m1 = pair[1];
        if (m0.distance >= static_cast<float>(ratio_test_ * m1.distance)) {
            continue;
        }
        if (!used_train.insert(m0.trainIdx).second) {
            continue;
        }

        prev_pts.push_back(prev_kps[m0.queryIdx].pt);
        curr_guess.push_back(curr_kps[m0.trainIdx].pt);
    }

    out.raw_matches = static_cast<int>(prev_pts.size());
    if (static_cast<int>(prev_pts.size()) < min_matches_) {
        return false;
    }

    std::vector<cv::Point2f> curr_lk = curr_guess;
    std::vector<uchar> status_fwd;
    std::vector<float> err_fwd;
    cv::calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts,
        curr_lk,
        status_fwd,
        err_fwd,
        cv::Size(lk_win_size_, lk_win_size_),
        lk_max_level_,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_USE_INITIAL_FLOW);

    std::vector<cv::Point2f> prev_back;
    std::vector<uchar> status_bwd;
    std::vector<float> err_bwd;
    cv::calcOpticalFlowPyrLK(
        curr_gray,
        prev_gray,
        curr_lk,
        prev_back,
        status_bwd,
        err_bwd,
        cv::Size(lk_win_size_, lk_win_size_),
        lk_max_level_,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01));

    const cv::Size image_size = curr_gray.size();
    for (std::size_t i = 0; i < prev_pts.size(); ++i) {
        if (!status_fwd[i] || !status_bwd[i]) {
            continue;
        }
        if (!inside_image(curr_lk[i], image_size)) {
            continue;
        }

        const double fb = cv::norm(prev_pts[i] - prev_back[i]);
        if (!std::isfinite(fb) || fb > fb_max_error_px_) {
            continue;
        }

        out.prev_pts.push_back(prev_pts[i]);
        out.curr_pts.push_back(curr_lk[i]);
    }

    out.tracked_points = static_cast<int>(out.prev_pts.size());
    return out.tracked_points >= min_matches_;
}

} // namespace vo
