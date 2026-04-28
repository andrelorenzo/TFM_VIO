#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "config.hpp"

namespace vo {

struct TrackSet {
    std::vector<cv::Point2f> prev_pts;
    std::vector<cv::Point2f> curr_pts;
    int raw_matches = 0;
    int tracked_points = 0;
};

class OrbFlowTracker {
public:
    explicit OrbFlowTracker(const Config& config);

    bool track(const cv::Mat& prev_gray, const cv::Mat& curr_gray, TrackSet& out) const;

private:
    void detectAndDescribe(const cv::Mat& gray,
                       std::vector<cv::KeyPoint>& kps,
                       cv::Mat& desc,
                       int fast_threshold) const;

    cv::Ptr<cv::ORB> orb_;
    cv::BFMatcher matcher_;

    int nfeatures_ = 1200;
    int ini_th_fast_ = 20;
    int min_th_fast_ = 7;
    int min_matches_ = 40;
    double ratio_test_ = 0.75;
    int lk_win_size_ = 21;
    int lk_max_level_ = 3;
    double fb_max_error_px_ = 1.5;
    int retry_min_keypoints_ = 120;
};

} // namespace vo
