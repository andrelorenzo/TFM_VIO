#pragma once
#include "config.hpp"

#include <unordered_map>

struct FeatureTrack {
    int id = -1;
    int root_image_id = -1;
    int last_image_id = -1;
    bool active = false;
    std::vector<cv::Point2f> meas_px;
    std::vector<cv::Point2f> meas_un;
};

void featuresInit(Config * config);
void featuresReset();
void featuresUpdateMeasurements(int image_id,
                                const std::vector<int>& ids,
                                const std::vector<cv::Point2f>& pts_px,
                                const std::vector<cv::Point2f>& pts_un);
void featuresRemove(const std::vector<int>& ids);
bool featuresReanchor(int id,
                      int image_id,
                      const cv::Point2f& pt_px,
                      const cv::Point2f& pt_un);
void featuresPrune(int min_last_image_id);
bool featuresGetTrack(int id, FeatureTrack* out);
const std::unordered_map<int, FeatureTrack>& featuresMap();
std::vector<int> featuresActiveIds();
