#include "features.hpp"

#include <algorithm>

static Config g_features_cfg;
static bool g_features_init = false;
static std::unordered_map<int, FeatureTrack> g_tracks;

void featuresInit(Config * config){
    if (config == nullptr) return;
    g_features_cfg = *config;
    featuresReset();
    g_features_init = true;
}

void featuresReset() {
    g_tracks.clear();
}

void featuresUpdateMeasurements(int image_id, const std::vector<int>& ids, const std::vector<cv::Point2f>& pts_px, const std::vector<cv::Point2f>& pts_un) {
    if (!g_features_init) return;
    if (ids.size() != pts_px.size() || ids.size() != pts_un.size()) return;

    for (size_t i = 0; i < ids.size(); ++i) {
        FeatureTrack& tr = g_tracks[ids[i]];
        if (tr.id < 0) {
            tr.id = ids[i];
            tr.root_image_id = image_id;
            tr.last_image_id = image_id;
            tr.active = true;
            tr.meas_px.clear();
            tr.meas_un.clear();
        }

        tr.active = true;
        tr.last_image_id = image_id;
        tr.meas_px.push_back(pts_px[i]);
        tr.meas_un.push_back(pts_un[i]);
    }
}

void featuresRemove(const std::vector<int>& ids) {
    for (int id : ids) {
        auto it = g_tracks.find(id);
        if (it != g_tracks.end()) {
            it->second.active = false;
        }
    }
}

bool featuresReanchor(int id, int image_id, const cv::Point2f& pt_px, const cv::Point2f& pt_un) {
    auto it = g_tracks.find(id);
    if (it == g_tracks.end()) return false;

    FeatureTrack& tr = it->second;
    tr.root_image_id = image_id;
    tr.last_image_id = image_id;
    tr.active = true;
    tr.meas_px.clear();
    tr.meas_un.clear();
    tr.meas_px.push_back(pt_px);
    tr.meas_un.push_back(pt_un);
    return true;
}

void featuresPrune(int min_last_image_id) {
    for (auto it = g_tracks.begin(); it != g_tracks.end(); ) {
        if (!it->second.active && it->second.last_image_id < min_last_image_id) {
            it = g_tracks.erase(it);
        } else {
            ++it;
        }
    }
}

bool featuresGetTrack(int id, FeatureTrack* out) {
    if (out == nullptr) return false;
    auto it = g_tracks.find(id);
    if (it == g_tracks.end()) return false;
    *out = it->second;
    return true;
}

const std::unordered_map<int, FeatureTrack>& featuresMap() {
    return g_tracks;
}

std::vector<int> featuresActiveIds() {
    std::vector<int> ids;
    ids.reserve(g_tracks.size());
    for (const auto& kv : g_tracks) {
        if (kv.second.active) ids.push_back(kv.first);
    }
    return ids;
}
