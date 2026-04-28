#pragma once
#include "config.hpp"

#include <vector>

struct TrackerOutput {
    int image_id = -1;
    bool first_frame = false;

    std::vector<int> tracked_ids;
    std::vector<int> tracked_track_len;
    std::vector<cv::Point2f> tracked_prev_px;
    std::vector<cv::Point2f> tracked_prev_un;
    std::vector<cv::Point2f> tracked_px;
    std::vector<cv::Point2f> tracked_un;

    std::vector<int> new_ids;
    std::vector<int> new_track_len;
    std::vector<cv::Point2f> new_px;
    std::vector<cv::Point2f> new_un;

    std::vector<int> lost_ids;
    std::vector<int> lost_track_len;
};

void trackerInit(Config * config);
void trackerReset();
bool trackerTrackFrame(const cv::Mat& frame_bgr, TrackerOutput* out);
