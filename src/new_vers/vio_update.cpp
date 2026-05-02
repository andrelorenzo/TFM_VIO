#include "vio_update.hpp"
#include "tracker.hpp"
#include "lie_math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace {

struct KeyframeState {
    bool valid = false;
    int image_id = -1;
    double ts_ms = 0.0;
    quat q_wc = quat::Identity();
    vec3 p_w = vec3::Zero();
    bool imu_pose_valid = false;
    quat q_wc_imu = quat::Identity();
    vec3 p_w_imu = vec3::Zero();
    std::unordered_map<int, cv::Point2f> px_by_id;
    std::unordered_map<int, cv::Point2f> un_by_id;
};

struct LandmarkState {
    int id = -1;
    cv::Point2f anchor_un = cv::Point2f(0.0f, 0.0f);
    double rho = 0.0;
    vec3 p_w = vec3::Zero();
    int obs_count = 0;
    int anchor_image_id = -1;
    int last_image_id = -1;
    double last_reproj_px = 0.0;
};

struct MatchBuckets {
    std::vector<int> init_slam_idx;
    std::vector<int> pose_only_idx;
    std::vector<int> exploration_idx;
};

struct RefSelection {
    const KeyframeState* kf = nullptr;
    int ref_age = 0;
    double raw_parallax_deg = 0.0;
    std::vector<int> ids;
    std::vector<cv::Point2f> kf_px;
    std::vector<cv::Point2f> kf_un;
    std::vector<cv::Point2f> cur_px;
    std::vector<cv::Point2f> cur_un;
    MatchBuckets buckets;
};

struct LocalBaObs {
    const KeyframeState* kf = nullptr;
    cv::Point2f px = cv::Point2f(0.0f, 0.0f);
};

struct LocalBaLandmark {
    int id = -1;
    LandmarkState* lm = nullptr;
    cv::Point2f current_px = cv::Point2f(0.0f, 0.0f);
    std::vector<LocalBaObs> fixed_obs;
};

struct ViFusionState {
    bool aligned = false;
    quat q_fuse_from_vis = quat::Identity();
    vec3 p_fuse_from_vis = vec3::Zero();
    bool prev_vis_valid = false;
    vec3 prev_vis_pos = vec3::Zero();
    double prev_vis_ts_ms = 0.0;
};

static Config g_vio_cfg;
static bool g_vio_ready = false;
static quat g_vis_q = quat::Identity();
static vec3 g_vis_p = vec3::Zero();
static std::deque<KeyframeState> g_kf_window;
static std::unordered_map<int, LandmarkState> g_landmarks;
static int g_frames_from_kf = 0;
static ViFusionState g_vi_fuse;
static bool g_inc_vo_ready = false;
static vec3 g_inc_prev_imu_p = vec3::Zero();
static quat g_inc_prev_imu_q = quat::Identity();
static double g_inc_prev_ts_ms = 0.0;
static mat3 g_R_ic = mat3::Identity();
static vec3 g_t_ic = vec3::Zero();
static mat3 g_R_ci = mat3::Identity();
static vec3 g_t_ci = vec3::Zero();
static bool g_have_cam_imu_extr = false;
static bool g_curr_imu_cam_pose_valid = false;
static quat g_curr_imu_cam_q = quat::Identity();
static vec3 g_curr_imu_cam_p = vec3::Zero();

static cv::Mat normalizeTransform4x4(const cv::Mat& input) {
    if (input.empty()) return cv::Mat();

    cv::Mat output = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat source;
    input.convertTo(source, CV_64F);
    if (source.rows < 3 || source.cols < 4) return cv::Mat();

    source(cv::Range(0, 3), cv::Range(0, 4)).copyTo(output(cv::Range(0, 3), cv::Range(0, 4)));
    return output;
}

static cv::Mat invertRigidTransform4x4(const cv::Mat& input) {
    const cv::Mat T = normalizeTransform4x4(input);
    if (T.empty()) return cv::Mat();

    const cv::Mat R = T(cv::Range(0, 3), cv::Range(0, 3)).clone();
    const cv::Mat t = T(cv::Range(0, 3), cv::Range(3, 4)).clone();
    const cv::Mat R_inv = R.t();
    const cv::Mat t_inv = -R_inv * t;

    cv::Mat output = cv::Mat::eye(4, 4, CV_64F);
    R_inv.copyTo(output(cv::Range(0, 3), cv::Range(0, 3)));
    t_inv.copyTo(output(cv::Range(0, 3), cv::Range(3, 4)));
    return output;
}

static void loadImuCameraExtrinsics() {
    g_R_ic = mat3::Identity();
    g_t_ic = vec3::Zero();
    g_R_ci = mat3::Identity();
    g_t_ci = vec3::Zero();
    g_have_cam_imu_extr = false;

    const cv::Mat T_raw = normalizeTransform4x4(g_vio_cfg.imu.T);
    if (T_raw.empty()) return;

    const cv::Mat T_ic = invertRigidTransform4x4(T_raw);
    if (T_ic.empty()) return;

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            g_R_ic(r, c) = T_ic.at<double>(r, c);
        }
        g_t_ic(r) = T_ic.at<double>(r, 3);
    }

    g_R_ci = g_R_ic.transpose();
    g_t_ci = -g_R_ci * g_t_ic;
    g_have_cam_imu_extr = true;
}

static bool cameraPoseFromImuPose(const quat& q_wi,
                                  const vec3& p_iw,
                                  quat* q_wc,
                                  vec3* p_cw) {
    if (q_wc == nullptr || p_cw == nullptr) return false;
    if (!q_wi.coeffs().allFinite() || !p_iw.allFinite()) return false;

    const mat3 R_wi = normalizeQ(q_wi).toRotationMatrix();
    const mat3 R_wc = R_wi * g_R_ic;
    *q_wc = normalizeQ(quat(R_wc));
    *p_cw = p_iw + R_wi * g_t_ic;
    return q_wc->coeffs().allFinite() && p_cw->allFinite();
}

static bool imuPoseFromCameraPose(const quat& q_wc,
                                  const vec3& p_cw,
                                  quat* q_wi,
                                  vec3* p_iw) {
    if (q_wi == nullptr || p_iw == nullptr) return false;
    if (!q_wc.coeffs().allFinite() || !p_cw.allFinite()) return false;

    const mat3 R_wc = normalizeQ(q_wc).toRotationMatrix();
    const mat3 R_wi = R_wc * g_R_ci;
    *q_wi = normalizeQ(quat(R_wi));
    *p_iw = p_cw - R_wi * g_t_ic;
    return q_wi->coeffs().allFinite() && p_iw->allFinite();
}

static bool imuCameraPoseFromState(const StateOut& state, quat* q_wc, vec3* p_cw) {
    if (!g_vio_cfg.gen.imu_on) return false;
    return cameraPoseFromImuPose(state.quat_rad, state.pos_m, q_wc, p_cw);
}

static void setVisualDebugFromCameraPose(const quat& q_wc, const vec3& p_cw, DebugState* deb) {
    if (deb == nullptr) return;

    quat q_wi = quat::Identity();
    vec3 p_iw = vec3::Zero();
    if (imuPoseFromCameraPose(q_wc, p_cw, &q_wi, &p_iw)) {
        deb->vis_xyz = p_iw;
        deb->vis_rpy = quatToCameraRpyRad(q_wi);
        return;
    }

    deb->vis_xyz = p_cw;
    deb->vis_rpy = quatToCameraRpyRad(q_wc);
}

static void updateCurrentImuCameraPose(const StateOut& state) {
    g_curr_imu_cam_pose_valid = imuCameraPoseFromState(state, &g_curr_imu_cam_q, &g_curr_imu_cam_p);
    if (!g_curr_imu_cam_pose_valid) {
        g_curr_imu_cam_q = quat::Identity();
        g_curr_imu_cam_p = vec3::Zero();
    }
}

static double normalizedRansacThreshold() {
    if (g_vio_cfg.cam.K.empty()) {
        return std::max(1e-4, g_vio_cfg.trk.ransac_f_px);
    }

    const double fx = g_vio_cfg.cam.K.at<double>(0, 0);
    const double fy = g_vio_cfg.cam.K.at<double>(1, 1);
    const double fmean = std::max(1.0, 0.5 * (std::abs(fx) + std::abs(fy)));
    return std::max(1e-4, g_vio_cfg.trk.ransac_f_px / fmean);
}

static const KeyframeState* latestKeyframe() {
    return g_kf_window.empty() ? nullptr : &g_kf_window.back();
}

static const KeyframeState* findKeyframeById(int image_id) {
    for (const KeyframeState& kf : g_kf_window) {
        if (kf.image_id == image_id) return &kf;
    }
    return nullptr;
}

static bool keyframeInWindow(int image_id) {
    return findKeyframeById(image_id) != nullptr;
}

static double elapsedSecondsFromKeyframe(const KeyframeState& ref_kf, double frame_ts_ms) {
    const double dt_ms = frame_ts_ms - ref_kf.ts_ms;
    if (std::isfinite(dt_ms) && dt_ms > 1e-3) {
        return 1e-3 * dt_ms;
    }

    if (g_vio_cfg.cam.fps > 1e-6) {
        return std::max(1, g_frames_from_kf + 1) / g_vio_cfg.cam.fps;
    }
    return std::max(1, g_frames_from_kf + 1);
}

static double computeStepMetersFromKeyframe(const KeyframeState& ref_kf, double frame_ts_ms) {
    return std::max(0.0, g_vio_cfg.vio.visual_step_scale) * elapsedSecondsFromKeyframe(ref_kf, frame_ts_ms);
}

static cv::Point2f normToPixel(const cv::Point2f& un) {
    if (g_vio_cfg.cam.K.empty()) return un;
    const double fx = g_vio_cfg.cam.K.at<double>(0, 0);
    const double fy = g_vio_cfg.cam.K.at<double>(1, 1);
    const double cx = g_vio_cfg.cam.K.at<double>(0, 2);
    const double cy = g_vio_cfg.cam.K.at<double>(1, 2);
    return cv::Point2f(static_cast<float>(fx * un.x + cx), static_cast<float>(fy * un.y + cy));
}

static double computeStepMetersFromState(const KeyframeState& ref_kf, const StateOut& state, double frame_ts_ms) {
    if (g_curr_imu_cam_pose_valid && ref_kf.imu_pose_valid) {
        const double imu_step = (g_curr_imu_cam_p - ref_kf.p_w_imu).norm();
        if (std::isfinite(imu_step) && imu_step > 1e-5) return imu_step;
    }
    return computeStepMetersFromKeyframe(ref_kf, frame_ts_ms);
}

static vec3 rotateRefBearingToCurrent(const cv::Point2f& ref_un,
                                      const quat& q_ref_wc,
                                      const quat& q_cur_wc) {
    vec3 b_ref(ref_un.x, ref_un.y, 1.0);
    b_ref.normalize();
    const mat3 R_cur_ref =
        normalizeQ(q_cur_wc).toRotationMatrix().transpose() *
        normalizeQ(q_ref_wc).toRotationMatrix();
    vec3 b_cur = R_cur_ref * b_ref;
    if (b_cur.norm() > 1e-12) b_cur.normalize();
    return b_cur;
}

static double medianParallaxDeg(const std::vector<cv::Point2f>& ref_un,
                                const std::vector<cv::Point2f>& cur_un,
                                const cv::Mat& mask,
                                const quat* q_ref_wc = nullptr,
                                const quat* q_cur_wc = nullptr) {
    std::vector<double> angles_deg;
    const uchar* mask_ptr = mask.empty() ? nullptr : mask.ptr<uchar>(0);

    for (size_t i = 0; i < ref_un.size() && i < cur_un.size(); ++i) {
        if (mask_ptr != nullptr && mask_ptr[i] == 0U) continue;

        vec3 b0(ref_un[i].x, ref_un[i].y, 1.0);
        vec3 b1(cur_un[i].x, cur_un[i].y, 1.0);
        if (q_ref_wc != nullptr && q_cur_wc != nullptr) {
            b0 = rotateRefBearingToCurrent(ref_un[i], *q_ref_wc, *q_cur_wc);
        } else {
            b0.normalize();
        }
        b1.normalize();
        const double c = std::clamp(b0.dot(b1), -1.0, 1.0);
        angles_deg.push_back(std::acos(c) * 180.0 / M_PI);
    }

    if (angles_deg.empty()) return 0.0;
    const size_t mid = angles_deg.size() / 2;
    std::nth_element(angles_deg.begin(), angles_deg.begin() + mid, angles_deg.end());
    return angles_deg[mid];
}

static double parallaxDeg(const cv::Point2f& ref_un,
                          const cv::Point2f& cur_un,
                          const quat* q_ref_wc = nullptr,
                          const quat* q_cur_wc = nullptr) {
    vec3 b0(ref_un.x, ref_un.y, 1.0);
    vec3 b1(cur_un.x, cur_un.y, 1.0);
    if (q_ref_wc != nullptr && q_cur_wc != nullptr) {
        b0 = rotateRefBearingToCurrent(ref_un, *q_ref_wc, *q_cur_wc);
    } else {
        b0.normalize();
    }
    b1.normalize();
    const double c = std::clamp(b0.dot(b1), -1.0, 1.0);
    return std::acos(c) * 180.0 / M_PI;
}

static std::unordered_map<int, int> buildTrackLengthMap(const TrackerOutput& track) {
    std::unordered_map<int, int> out;
    out.reserve(track.tracked_ids.size() + track.new_ids.size() + track.lost_ids.size());

    for (size_t i = 0; i < track.tracked_ids.size(); ++i) {
        const int len = (i < track.tracked_track_len.size()) ? track.tracked_track_len[i] : 0;
        out[track.tracked_ids[i]] = len;
    }
    for (size_t i = 0; i < track.new_ids.size(); ++i) {
        const int len = (i < track.new_track_len.size()) ? track.new_track_len[i] : 0;
        out[track.new_ids[i]] = len;
    }
    for (size_t i = 0; i < track.lost_ids.size(); ++i) {
        const int len = (i < track.lost_track_len.size()) ? track.lost_track_len[i] : 0;
        out[track.lost_ids[i]] = len;
    }
    return out;
}

static void collectCurrentMeasurements(const TrackerOutput& track,
                                       std::unordered_map<int, cv::Point2f>* px_by_id,
                                       std::unordered_map<int, cv::Point2f>* un_by_id) {
    if (px_by_id == nullptr || un_by_id == nullptr) return;

    px_by_id->clear();
    un_by_id->clear();

    for (size_t i = 0; i < track.tracked_ids.size() && i < track.tracked_px.size() && i < track.tracked_un.size(); ++i) {
        (*px_by_id)[track.tracked_ids[i]] = track.tracked_px[i];
        (*un_by_id)[track.tracked_ids[i]] = track.tracked_un[i];
    }

    for (size_t i = 0; i < track.new_ids.size() && i < track.new_px.size() && i < track.new_un.size(); ++i) {
        (*px_by_id)[track.new_ids[i]] = track.new_px[i];
        (*un_by_id)[track.new_ids[i]] = track.new_un[i];
    }
}

static void touchTrackedLandmarks(const TrackerOutput& track) {
    for (int id : track.tracked_ids) {
        auto it = g_landmarks.find(id);
        if (it == g_landmarks.end()) continue;
        LandmarkState& lm = it->second;
        if (lm.last_image_id != track.image_id) {
            lm.obs_count = std::max(lm.obs_count, 1) + 1;
        }
        lm.last_image_id = track.image_id;
    }
}

static void setKeyframeFromTrack(const TrackerOutput& track,
                                 double ts_ms,
                                 const quat& q_wc,
                                 const vec3& p_w,
                                 const quat* q_wc_imu = nullptr,
                                 const vec3* p_w_imu = nullptr) {
    KeyframeState kf;
    kf.valid = true;
    kf.image_id = track.image_id;
    kf.ts_ms = ts_ms;
    kf.q_wc = normalizeQ(q_wc);
    kf.p_w = p_w;
    if (q_wc_imu != nullptr && p_w_imu != nullptr && q_wc_imu->coeffs().allFinite() && p_w_imu->allFinite()) {
        kf.imu_pose_valid = true;
        kf.q_wc_imu = normalizeQ(*q_wc_imu);
        kf.p_w_imu = *p_w_imu;
    }
    collectCurrentMeasurements(track, &kf.px_by_id, &kf.un_by_id);
    g_kf_window.push_back(std::move(kf));
    while (static_cast<int>(g_kf_window.size()) > std::max(1, g_vio_cfg.vio.window_size)) {
        g_kf_window.pop_front();
    }
    g_frames_from_kf = 0;
}

static void resetVisualMapToState(const TrackerOutput& track, double ts_ms, StateOut* state, const char* reason) {
    if (state == nullptr) return;
    quat imu_cam_q = quat::Identity();
    vec3 imu_cam_p = vec3::Zero();
    if (!imuCameraPoseFromState(*state, &imu_cam_q, &imu_cam_p)) {
        imu_cam_q = quat::Identity();
        imu_cam_p = vec3::Zero();
    }

    g_vis_q = imu_cam_q;
    g_vis_p = imu_cam_p;
    g_kf_window.clear();
    g_landmarks.clear();
    g_frames_from_kf = 0;
    g_vi_fuse = ViFusionState{};
    g_inc_vo_ready = false;
    g_inc_prev_imu_p = vec3::Zero();
    g_inc_prev_imu_q = quat::Identity();
    g_inc_prev_ts_ms = 0.0;
    setKeyframeFromTrack(track, ts_ms, g_vis_q, g_vis_p, &imu_cam_q, &imu_cam_p);
    setVisualDebugFromCameraPose(g_vis_q, g_vis_p, &state->deb);
    state->deb.vio_window_kfs = static_cast<int>(g_kf_window.size());
    state->deb.vio_landmarks = 0;
    Logger(WARN, "[VIO_RESET] reason=%s img=%d ts=%.3f p=[%.3f %.3f %.3f] rpy=[%.3f %.3f %.3f]",
           reason != nullptr ? reason : "?", track.image_id, ts_ms,
           g_vis_p.x(), g_vis_p.y(), g_vis_p.z(), state->deb.vis_rpy.x(), state->deb.vis_rpy.y(), state->deb.vis_rpy.z());
}

static void extractKeyframeMatches(const TrackerOutput& track,
                                   const KeyframeState& ref_kf,
                                   std::vector<int>* ids,
                                   std::vector<cv::Point2f>* kf_px,
                                   std::vector<cv::Point2f>* kf_un,
                                   std::vector<cv::Point2f>* cur_px,
                                   std::vector<cv::Point2f>* cur_un) {
    if (ids == nullptr || kf_px == nullptr || kf_un == nullptr || cur_px == nullptr || cur_un == nullptr) {
        return;
    }

    ids->clear();
    kf_px->clear();
    kf_un->clear();
    cur_px->clear();
    cur_un->clear();

    for (size_t i = 0; i < track.tracked_ids.size() && i < track.tracked_px.size() && i < track.tracked_un.size(); ++i) {
        const int id = track.tracked_ids[i];
        const auto it_px = ref_kf.px_by_id.find(id);
        const auto it_un = ref_kf.un_by_id.find(id);
        if (it_px == ref_kf.px_by_id.end() || it_un == ref_kf.un_by_id.end()) continue;

        ids->push_back(id);
        kf_px->push_back(it_px->second);
        kf_un->push_back(it_un->second);
        cur_px->push_back(track.tracked_px[i]);
        cur_un->push_back(track.tracked_un[i]);
    }
}

static bool landmarkWorldPoint(const LandmarkState& lm, vec3* p_w) {
    if (p_w == nullptr) return false;
    const KeyframeState* anchor_kf = findKeyframeById(lm.anchor_image_id);
    if (anchor_kf == nullptr) return false;
    if (!std::isfinite(lm.rho) || lm.rho <= 1e-9) return false;

    const vec3 p_anchor(
        static_cast<double>(lm.anchor_un.x) / lm.rho,
        static_cast<double>(lm.anchor_un.y) / lm.rho,
        1.0 / lm.rho);
    *p_w = anchor_kf->q_wc.toRotationMatrix() * p_anchor + anchor_kf->p_w;
    return p_w->allFinite();
}

static std::vector<LocalBaLandmark> collectLocalBaLandmarks(const TrackerOutput& track) {
    struct Candidate {
        LocalBaLandmark lm;
        double score = 0.0;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(track.tracked_ids.size());

    for (size_t i = 0; i < track.tracked_ids.size() && i < track.tracked_un.size(); ++i) {
        const int id = track.tracked_ids[i];
        auto it = g_landmarks.find(id);
        if (it == g_landmarks.end()) continue;

        LandmarkState& lm = it->second;
        if (lm.obs_count < g_vio_cfg.vio.landmark_min_obs) continue;
        if (lm.last_reproj_px > g_vio_cfg.vio.landmark_max_reproj_px) continue;
        if (!keyframeInWindow(lm.anchor_image_id)) continue;

        vec3 p_w = vec3::Zero();
        if (!landmarkWorldPoint(lm, &p_w)) continue;

        Candidate cand;
        cand.lm.id = id;
        cand.lm.lm = &lm;
        cand.lm.current_px = normToPixel(track.tracked_un[i]);

        for (const KeyframeState& kf : g_kf_window) {
            if (kf.image_id == lm.anchor_image_id) continue;
            const auto it_un = kf.un_by_id.find(id);
            if (it_un == kf.un_by_id.end()) continue;
            cand.lm.fixed_obs.push_back(LocalBaObs{&kf, normToPixel(it_un->second)});
        }

        if (cand.lm.fixed_obs.empty()) continue;

        cand.score =
            10.0 * static_cast<double>(lm.obs_count) -
            2.0 * lm.last_reproj_px -
            0.05 * static_cast<double>(track.image_id - lm.last_image_id);
        candidates.push_back(std::move(cand));
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    std::vector<LocalBaLandmark> out;
    const int max_landmarks = std::max(1, g_vio_cfg.vio.local_ba_max_landmarks);
    for (int i = 0; i < static_cast<int>(candidates.size()) && i < max_landmarks; ++i) {
        out.push_back(std::move(candidates[i].lm));
    }
    return out;
}

static MatchBuckets classifyKeyframeMatches(const KeyframeState& ref_kf,
                                            const std::vector<int>& ids,
                                            const std::vector<cv::Point2f>& kf_un,
                                            const std::vector<cv::Point2f>& cur_un,
                                            const std::unordered_map<int, int>& track_len_by_id) {
    MatchBuckets buckets;
    const quat* q_ref = ref_kf.imu_pose_valid ? &ref_kf.q_wc_imu : nullptr;
    const quat* q_cur = g_curr_imu_cam_pose_valid ? &g_curr_imu_cam_q : nullptr;

    for (size_t i = 0; i < ids.size() && i < kf_un.size() && i < cur_un.size(); ++i) {
        const int id = ids[i];
        const auto it_len = track_len_by_id.find(id);
        const int track_len = (it_len != track_len_by_id.end()) ? it_len->second : 0;
        const double para_deg = parallaxDeg(kf_un[i], cur_un[i], q_ref, q_cur);

        const auto it_lm = g_landmarks.find(id);
        const bool has_landmark =
            it_lm != g_landmarks.end() &&
            keyframeInWindow(it_lm->second.anchor_image_id) &&
            it_lm->second.p_w.allFinite() &&
            it_lm->second.obs_count >= g_vio_cfg.vio.landmark_min_obs &&
            it_lm->second.last_reproj_px <= g_vio_cfg.vio.landmark_max_reproj_px;

        if (has_landmark) {
            buckets.exploration_idx.push_back(static_cast<int>(i));
        } else if (track_len >= g_vio_cfg.vio.min_track_length && para_deg >= g_vio_cfg.vio.good_parallax_deg) {
            buckets.init_slam_idx.push_back(static_cast<int>(i));
        } else if (track_len >= g_vio_cfg.vio.min_track_length) {
            buckets.pose_only_idx.push_back(static_cast<int>(i));
        }
    }

    return buckets;
}

static void selectMatchesByIndex(const std::vector<int>& idx,
                                 const std::vector<int>& src_ids,
                                 const std::vector<cv::Point2f>& src_kf_px,
                                 const std::vector<cv::Point2f>& src_kf_un,
                                 const std::vector<cv::Point2f>& src_cur_px,
                                 const std::vector<cv::Point2f>& src_cur_un,
                                 std::vector<int>* dst_ids,
                                 std::vector<cv::Point2f>* dst_kf_px,
                                 std::vector<cv::Point2f>* dst_kf_un,
                                 std::vector<cv::Point2f>* dst_cur_px,
                                 std::vector<cv::Point2f>* dst_cur_un) {
    if (dst_ids == nullptr || dst_kf_px == nullptr || dst_kf_un == nullptr ||
        dst_cur_px == nullptr || dst_cur_un == nullptr) {
        return;
    }

    dst_ids->clear();
    dst_kf_px->clear();
    dst_kf_un->clear();
    dst_cur_px->clear();
    dst_cur_un->clear();

    dst_ids->reserve(idx.size());
    dst_kf_px->reserve(idx.size());
    dst_kf_un->reserve(idx.size());
    dst_cur_px->reserve(idx.size());
    dst_cur_un->reserve(idx.size());

    for (const int k : idx) {
        if (k < 0 || k >= static_cast<int>(src_ids.size()) ||
            k >= static_cast<int>(src_kf_px.size()) ||
            k >= static_cast<int>(src_kf_un.size()) ||
            k >= static_cast<int>(src_cur_px.size()) ||
            k >= static_cast<int>(src_cur_un.size())) {
            continue;
        }

        dst_ids->push_back(src_ids[k]);
        dst_kf_px->push_back(src_kf_px[k]);
        dst_kf_un->push_back(src_kf_un[k]);
        dst_cur_px->push_back(src_cur_px[k]);
        dst_cur_un->push_back(src_cur_un[k]);
    }
}

static bool chooseReferenceKeyframe(const TrackerOutput& track,
                                    const std::unordered_map<int, int>& track_len_by_id,
                                    RefSelection* out) {
    if (out == nullptr) return false;
    *out = RefSelection{};

    double best_score = -1.0;
    for (auto it = g_kf_window.rbegin(); it != g_kf_window.rend(); ++it) {
        RefSelection cand;
        cand.kf = &(*it);
        cand.ref_age = std::max(0, track.image_id - it->image_id);

        extractKeyframeMatches(track, *it, &cand.ids, &cand.kf_px, &cand.kf_un, &cand.cur_px, &cand.cur_un);
        if (static_cast<int>(cand.ids.size()) < std::max(5, g_vio_cfg.trk.min_ransac_points)) continue;

        cand.raw_parallax_deg = medianParallaxDeg(
            cand.kf_un,
            cand.cur_un,
            cv::Mat(),
            it->imu_pose_valid ? &it->q_wc_imu : nullptr,
            g_curr_imu_cam_pose_valid ? &g_curr_imu_cam_q : nullptr);
        cand.buckets = classifyKeyframeMatches(*it, cand.ids, cand.kf_un, cand.cur_un, track_len_by_id);

        const double score =
            static_cast<double>(cand.ids.size()) +
            1.5 * static_cast<double>(cand.buckets.init_slam_idx.size()) +
            0.5 * static_cast<double>(cand.buckets.exploration_idx.size()) +
            10.0 * std::min(cand.raw_parallax_deg, g_vio_cfg.vio.good_parallax_deg) -
            0.25 * static_cast<double>(cand.ref_age);

        if (score > best_score) {
            best_score = score;
            *out = std::move(cand);
        }
    }

    return out->kf != nullptr;
}

static bool estimateRelativeVisualPose(const std::vector<cv::Point2f>& ref_un,
                                       const std::vector<cv::Point2f>& cur_un,
                                       mat3* R_ref_curr,
                                       vec3* t_ref_curr_dir,
                                       int* n_corr,
                                       int* n_inliers,
                                       int* n_pose_inliers,
                                       double* median_parallax_deg) {
    if (R_ref_curr == nullptr || t_ref_curr_dir == nullptr || n_corr == nullptr ||
        n_inliers == nullptr || n_pose_inliers == nullptr || median_parallax_deg == nullptr) {
        return false;
    }

    *R_ref_curr = mat3::Identity();
    *t_ref_curr_dir = vec3::Zero();
    *n_corr = static_cast<int>(std::min(ref_un.size(), cur_un.size()));
    *n_inliers = 0;
    *n_pose_inliers = 0;
    *median_parallax_deg = 0.0;

    if (*n_corr < std::max(5, g_vio_cfg.trk.min_ransac_points)) return false;

    cv::Mat e_mask;
    const cv::Mat E = cv::findEssentialMat(
        ref_un,
        cur_un,
        1.0,
        cv::Point2d(0.0, 0.0),
        cv::RANSAC,
        std::clamp(g_vio_cfg.trk.ransac_conf, 0.5, 0.9999),
        normalizedRansacThreshold(),
        e_mask);
    if (E.empty()) return false;

    *n_inliers = cv::countNonZero(e_mask);
    if (*n_inliers < std::max(5, g_vio_cfg.trk.min_ransac_points)) return false;

    cv::Mat R_cv, t_cv;
    const int ninl_pose = cv::recoverPose(
        E,
        ref_un,
        cur_un,
        cv::Mat::eye(3, 3, CV_64F),
        R_cv,
        t_cv,
        e_mask);
    if (ninl_pose < std::max(5, g_vio_cfg.trk.min_ransac_points)) return false;

    *n_pose_inliers = ninl_pose;
    *median_parallax_deg = medianParallaxDeg(ref_un, cur_un, e_mask);

    R_cv.convertTo(R_cv, CV_64F);
    t_cv.convertTo(t_cv, CV_64F);

    mat3 R_cur_ref = mat3::Identity();
    vec3 t_cur = vec3::Zero();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_cur_ref(r, c) = R_cv.at<double>(r, c);
        }
        t_cur(r) = t_cv.at<double>(r, 0);
    }

    // recoverPose returns the transform from reference camera to current camera.
    // We keep camera-to-world poses, so we need the inverse rotation and the
    // current camera center expressed in the reference frame.
    *R_ref_curr = R_cur_ref.transpose();
    *t_ref_curr_dir = -(*R_ref_curr) * t_cur.normalized();
    return R_ref_curr->allFinite() && t_ref_curr_dir->allFinite() && t_ref_curr_dir->norm() > 1e-9;
}

static double visualScaleFromImuStep(const StateOut& state, double dt_ms) {
    const double dt_s = std::isfinite(dt_ms) && dt_ms > 0.0 ? 1e-3 * dt_ms : 0.0;
    double step = 0.0;

    if (g_inc_vo_ready && g_curr_imu_cam_pose_valid && g_inc_prev_imu_p.allFinite()) {
        step = (g_curr_imu_cam_p - g_inc_prev_imu_p).norm();
    }

    if ((!std::isfinite(step) || step < 1e-5) && state.deb.imu_dp.allFinite()) {
        step = state.deb.imu_dp.norm();
    }

    if ((!std::isfinite(step) || step < 1e-5) && state.vel_ms.allFinite() && dt_s > 0.0) {
        step = state.vel_ms.norm() * dt_s;
    }

    const double nominal_speed = std::max(0.05, g_vio_cfg.vio.visual_step_scale);
    const double max_step = std::max(0.02, 2.5 * nominal_speed * std::max(dt_s, 1e-3));

    if (!std::isfinite(step) || step < 0.0) step = 0.0;
    return std::clamp(step, 0.0, max_step);
}

static bool updateIncrementalVisualOdometry(const TrackerOutput& track, const StateOut& state, double frame_ts_ms, int* n_corr, int* n_inliers, int* n_pose_inliers, double* parallax_deg, double* step_m) {
    if (n_corr != nullptr) *n_corr = 0;
    if (n_inliers != nullptr) *n_inliers = 0;
    if (n_pose_inliers != nullptr) *n_pose_inliers = 0;
    if (parallax_deg != nullptr) *parallax_deg = 0.0;
    if (step_m != nullptr) *step_m = 0.0;

    if (!g_inc_vo_ready) {
        g_inc_vo_ready = true;
        g_inc_prev_imu_p = g_curr_imu_cam_pose_valid ? g_curr_imu_cam_p : vec3::Zero();
        g_inc_prev_imu_q = g_curr_imu_cam_pose_valid ? g_curr_imu_cam_q : quat::Identity();
        g_inc_prev_ts_ms = frame_ts_ms;
        return false;
    }

    mat3 R_prev_curr = mat3::Identity();
    vec3 t_prev_curr_dir = vec3::Zero();
    int nc = 0;
    int ni = 0;
    int npi = 0;
    double para = 0.0;

    const bool rel_ok = estimateRelativeVisualPose(track.tracked_prev_un, track.tracked_un, &R_prev_curr, &t_prev_curr_dir, &nc, &ni, &npi, &para);
    if (rel_ok && g_curr_imu_cam_pose_valid && g_inc_prev_imu_q.coeffs().allFinite()) {
        para = medianParallaxDeg(track.tracked_prev_un, track.tracked_un, cv::Mat(), &g_inc_prev_imu_q, &g_curr_imu_cam_q);
    }
    const double dt_ms = frame_ts_ms - g_inc_prev_ts_ms;
    const double scale = visualScaleFromImuStep(state, dt_ms);

    if (n_corr != nullptr) *n_corr = nc;
    if (n_inliers != nullptr) *n_inliers = ni;
    if (n_pose_inliers != nullptr) *n_pose_inliers = npi;
    if (parallax_deg != nullptr) *parallax_deg = para;
    if (step_m != nullptr) *step_m = scale;

    bool ok = false;
    if (rel_ok && scale > 0.0 && t_prev_curr_dir.norm() > 1e-9) {
        const quat q_prev = normalizeQ(g_vis_q);
        const vec3 p_prev = g_vis_p;
        const quat q_new = normalizeQ(q_prev * quat(R_prev_curr));
        const vec3 p_new = p_prev + q_prev.toRotationMatrix() * (scale * t_prev_curr_dir.normalized());

        if (q_new.coeffs().allFinite() && p_new.allFinite()) {
            g_vis_q = q_new;
            g_vis_p = p_new;
            ok = true;
        }
    }

    g_inc_prev_imu_p = g_curr_imu_cam_pose_valid ? g_curr_imu_cam_p : vec3::Zero();
    g_inc_prev_imu_q = g_curr_imu_cam_pose_valid ? g_curr_imu_cam_q : quat::Identity();
    g_inc_prev_ts_ms = frame_ts_ms;
    return ok;
}

static void worldPoseToExtrinsics(const quat& q_wc,
                                  const vec3& p_w,
                                  mat3* R_cw,
                                  vec3* t_cw) {
    const mat3 R_wc = q_wc.toRotationMatrix();
    if (R_cw != nullptr) {
        *R_cw = R_wc.transpose();
    }
    if (t_cw != nullptr) {
        *t_cw = -R_wc.transpose() * p_w;
    }
}

static cv::Mat projectionFromPose(const quat& q_wc, const vec3& p_w) {
    mat3 R_cw = mat3::Identity();
    vec3 t_cw = vec3::Zero();
    worldPoseToExtrinsics(q_wc, p_w, &R_cw, &t_cw);

    cv::Mat P = cv::Mat::zeros(3, 4, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            P.at<double>(r, c) = R_cw(r, c);
        }
        P.at<double>(r, 3) = t_cw(r);
    }
    return P;
}

static bool projectWorldPointPx(const vec3& p_w,
                                const quat& q_wc,
                                const vec3& cam_p_w,
                                cv::Point2f* px_out,
                                double* depth_out = nullptr) {
    if (px_out == nullptr) return false;

    mat3 R_cw = mat3::Identity();
    vec3 t_cw = vec3::Zero();
    worldPoseToExtrinsics(q_wc, cam_p_w, &R_cw, &t_cw);

    const vec3 p_c = R_cw * p_w + t_cw;
    if (!p_c.allFinite() || p_c.z() <= 1e-9) return false;

    if (depth_out != nullptr) {
        *depth_out = p_c.z();
    }

    const double x = p_c.x() / p_c.z();
    const double y = p_c.y() / p_c.z();

    if (g_vio_cfg.cam.K.empty()) {
        *px_out = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
        return true;
    }

    const double fx = g_vio_cfg.cam.K.at<double>(0, 0);
    const double fy = g_vio_cfg.cam.K.at<double>(1, 1);
    const double cx = g_vio_cfg.cam.K.at<double>(0, 2);
    const double cy = g_vio_cfg.cam.K.at<double>(1, 2);

    *px_out = cv::Point2f(
        static_cast<float>(fx * x + cx),
        static_cast<float>(fy * y + cy));
    return true;
}

static void buildPnPCorrespondences(const TrackerOutput& track,
                                    std::vector<cv::Point3f>* pts3,
                                    std::vector<cv::Point2f>* pts2,
                                    std::vector<int>* ids) {
    if (pts3 == nullptr || pts2 == nullptr || ids == nullptr) return;

    pts3->clear();
    pts2->clear();
    ids->clear();

    for (size_t i = 0; i < track.tracked_ids.size() && i < track.tracked_px.size(); ++i) {
        const int id = track.tracked_ids[i];
        const auto it = g_landmarks.find(id);
        if (it == g_landmarks.end()) continue;

        const LandmarkState& lm = it->second;
        const int track_len = (i < track.tracked_track_len.size()) ? track.tracked_track_len[i] : 0;
        const int age = (lm.last_image_id >= 0) ? (track.image_id - lm.last_image_id) : std::numeric_limits<int>::max();
        vec3 p_w = vec3::Zero();
        if (!landmarkWorldPoint(lm, &p_w)) continue;
        if (track_len < g_vio_cfg.vio.min_track_length) continue;
        if (lm.obs_count < g_vio_cfg.vio.landmark_min_obs) continue;
        if (age > g_vio_cfg.vio.landmark_max_age) continue;
        if (lm.last_reproj_px > g_vio_cfg.vio.landmark_max_reproj_px) continue;

        pts3->emplace_back(
            static_cast<float>(p_w.x()),
            static_cast<float>(p_w.y()),
            static_cast<float>(p_w.z()));
        pts2->push_back(track.tracked_px[i]);
        ids->push_back(id);
    }
}

static void buildPoseCorrespondencesUndistPx(const TrackerOutput& track, std::vector<cv::Point3f>* pts3, std::vector<cv::Point2f>* pts2, std::vector<int>* ids) {
    if (pts3 == nullptr || pts2 == nullptr || ids == nullptr) return;

    pts3->clear();
    pts2->clear();
    ids->clear();

    for (size_t i = 0; i < track.tracked_ids.size() && i < track.tracked_un.size(); ++i) {
        const int id = track.tracked_ids[i];
        const auto it = g_landmarks.find(id);
        if (it == g_landmarks.end()) continue;

        const LandmarkState& lm = it->second;
        const int track_len = (i < track.tracked_track_len.size()) ? track.tracked_track_len[i] : 0;
        const int age = (lm.last_image_id >= 0) ? (track.image_id - lm.last_image_id) : std::numeric_limits<int>::max();
        vec3 p_w = vec3::Zero();
        if (!landmarkWorldPoint(lm, &p_w)) continue;
        if (track_len < g_vio_cfg.vio.min_track_length) continue;
        if (lm.obs_count < g_vio_cfg.vio.landmark_min_obs) continue;
        if (age > g_vio_cfg.vio.landmark_max_age) continue;
        if (lm.last_reproj_px > g_vio_cfg.vio.landmark_max_reproj_px) continue;

        pts3->emplace_back(static_cast<float>(p_w.x()), static_cast<float>(p_w.y()), static_cast<float>(p_w.z()));
        pts2->push_back(normToPixel(track.tracked_un[i]));
        ids->push_back(id);
    }
}

static mat3 expSO3Std(const vec3& w) {
    const double theta = w.norm();
    if (theta < 1e-12) {
        return mat3::Identity() + skewMat(w);
    }
    return Eigen::AngleAxisd(theta, w / theta).toRotationMatrix();
}

static double quatDeltaDeg(const quat& q0, const quat& q1) {
    const quat dq = normalizeQ(q1 * q0.conjugate());
    const double w = std::clamp(std::abs(dq.w()), 0.0, 1.0);
    return 2.0 * std::acos(w) * 180.0 / M_PI;
}

static vec3 clampVecNorm(const vec3& v, double max_norm) {
    if (max_norm <= 0.0) return vec3::Zero();
    const double n = v.norm();
    if (!std::isfinite(n) || n <= max_norm) return v;
    return (max_norm / n) * v;
}

static double visualPoseMaxResidualM(const StateOut& state) {
    const double dt = std::isfinite(state.dt) && state.dt > 1e-4 ? state.dt : 0.10;
    const double v = state.vel_ms.allFinite() ? state.vel_ms.norm() : 0.0;
    const double dyn = 0.75 + 2.0 * v * dt;
    return std::clamp(dyn, 0.75, 8.0);
}

static bool visualPoseConsistent(const char* tag, const quat& q_wc, const vec3& p_w, const StateOut& state, double frame_ts_ms, int image_id) {
    if (!g_vio_cfg.gen.imu_on) return true;
    if (!p_w.allFinite() || !q_wc.coeffs().allFinite()) return false;
    if (!state.pos_m.allFinite() || !state.quat_rad.coeffs().allFinite()) return true;

    quat q_wi_vis = quat::Identity();
    vec3 p_iw_vis = vec3::Zero();
    if (!imuPoseFromCameraPose(q_wc, p_w, &q_wi_vis, &p_iw_vis)) return false;

    const double pos_err = (p_iw_vis - state.pos_m).norm();
    const double pos_max = visualPoseMaxResidualM(state);
    const double ori_err = quatDeltaDeg(state.quat_rad, q_wi_vis);
    const double ori_max = std::max(20.0, g_vio_cfg.vio.fuse_max_ori_corr_deg);

    if (!std::isfinite(pos_err) || pos_err > pos_max || !std::isfinite(ori_err) || ori_err > ori_max) {
        Logger(WARN, "[VIO_POSE_REJECT] tag=%s img=%d ts=%.3f pos_err=%.3f pos_max=%.3f ori_err=%.3f ori_max=%.3f visual_p=[%.3f %.3f %.3f] imu_p=[%.3f %.3f %.3f]",
               tag != nullptr ? tag : "?", image_id, frame_ts_ms, pos_err, pos_max, ori_err, ori_max,
               p_iw_vis.x(), p_iw_vis.y(), p_iw_vis.z(), state.pos_m.x(), state.pos_m.y(), state.pos_m.z());
        return false;
    }

    return true;
}

static bool visualOrientationConsistent(const char* tag, const quat& q_wc, const StateOut& state, double frame_ts_ms, int image_id) {
    if (!g_vio_cfg.gen.imu_on) return true;
    if (!q_wc.coeffs().allFinite()) return false;
    if (!state.quat_rad.coeffs().allFinite()) return true;

    const quat q_wi_vis = normalizeQ(quat(normalizeQ(q_wc).toRotationMatrix() * g_R_ci));
    const double ori_err = quatDeltaDeg(state.quat_rad, q_wi_vis);
    const double ori_max = std::max(20.0, g_vio_cfg.vio.fuse_max_ori_corr_deg);

    if (!std::isfinite(ori_err) || ori_err > ori_max) {
        Logger(WARN, "[VIO_ORI_REJECT] tag=%s img=%d ts=%.3f ori_err=%.3f ori_max=%.3f",
               tag != nullptr ? tag : "?", image_id, frame_ts_ms, ori_err, ori_max);
        return false;
    }

    return true;
}

static bool visualMapDiverged(const StateOut& state, double* err_out) {
    if (!g_vio_cfg.gen.imu_on) return false;
    if (!g_vis_p.allFinite() || !state.pos_m.allFinite()) return false;
    quat q_wi_vis = quat::Identity();
    vec3 p_iw_vis = vec3::Zero();
    if (!imuPoseFromCameraPose(g_vis_q, g_vis_p, &q_wi_vis, &p_iw_vis)) return false;
    const double err = (p_iw_vis - state.pos_m).norm();
    if (err_out != nullptr) *err_out = err;
    const double reset_thr = std::max(3.0, 3.0 * visualPoseMaxResidualM(state));
    return std::isfinite(err) && err > reset_thr;
}

static bool runVisualInertialCorrection(StateOut* state,
                                        bool visual_pose_ok,
                                        int visual_inliers,
                                        double visual_reproj_px,
                                        double visual_parallax_deg) {
    if (state == nullptr) return false;

    state->deb.vio_fused = false;
    state->deb.vio_fuse_pos_res_m = 0.0;
    state->deb.vio_fuse_vel_res_ms = 0.0;
    state->deb.vio_fuse_ori_res_deg = 0.0;

    if (!g_vio_cfg.vio.fuse_enable || !g_vio_cfg.gen.imu_on) return false;
    if (!visual_pose_ok) return false;
    if (visual_inliers < g_vio_cfg.vio.fuse_min_inliers) return false;
    if (!std::isfinite(visual_reproj_px) || visual_reproj_px > g_vio_cfg.vio.fuse_max_reproj_px) return false;
    if (!std::isfinite(visual_parallax_deg) || visual_parallax_deg < g_vio_cfg.vio.fuse_min_parallax_deg) return false;

    if (!g_vi_fuse.aligned) {
        g_vi_fuse.prev_vis_valid = false;
        g_vi_fuse.aligned = true;
    }

    const quat q_vis_cam = normalizeQ(g_vis_q);
    const vec3 p_vis_cam = g_vis_p;

    quat q_vis_in_fuse = quat::Identity();
    vec3 p_vis_in_fuse = vec3::Zero();
    if (!imuPoseFromCameraPose(q_vis_cam, p_vis_cam, &q_vis_in_fuse, &p_vis_in_fuse)) return false;

    if (!visualPoseConsistent("FUSE", q_vis_cam, p_vis_cam, *state, state->ts_ms, -1)) return false;

    const vec3 pos_res = p_vis_in_fuse - state->pos_m;
    const vec3 pos_corr = clampVecNorm(
        std::clamp(g_vio_cfg.vio.fuse_pos_gain, 0.0, 1.0) * pos_res,
        g_vio_cfg.vio.fuse_max_pos_corr_m);
    state->pos_m += pos_corr;
    state->deb.vio_fuse_pos_res_m = pos_res.norm();

    if (g_vi_fuse.prev_vis_valid) {
        const double dt_s = 1e-3 * (state->ts_ms - g_vi_fuse.prev_vis_ts_ms);
        if (std::isfinite(dt_s) && dt_s > 1e-3) {
            const vec3 vel_meas = (p_vis_in_fuse - g_vi_fuse.prev_vis_pos) / dt_s;
            const vec3 vel_res = vel_meas - state->vel_ms;
            const vec3 vel_corr = clampVecNorm(
                std::clamp(g_vio_cfg.vio.fuse_vel_gain, 0.0, 1.0) * vel_res,
                g_vio_cfg.vio.fuse_max_vel_corr_ms);
            state->vel_ms += vel_corr;
            state->deb.vio_fuse_vel_res_ms = vel_res.norm();
        }
    }

    const double ori_gain = std::clamp(g_vio_cfg.vio.fuse_ori_gain, 0.0, 1.0);
    const double ori_res_deg = quatDeltaDeg(state->quat_rad, q_vis_in_fuse);
    state->deb.vio_fuse_ori_res_deg = ori_res_deg;
    if (ori_gain > 0.0 && ori_res_deg <= g_vio_cfg.vio.fuse_max_ori_corr_deg) {
        const quat dq = normalizeQ(q_vis_in_fuse * state->quat_rad.conjugate());
        const quat q_corr = quat::Identity().slerp(ori_gain, dq);
        state->quat_rad = normalizeQ(q_corr * state->quat_rad);
        state->rpy_rad = quatToCameraRpyRad(state->quat_rad);
    }

    g_vi_fuse.prev_vis_pos = p_vis_in_fuse;
    g_vi_fuse.prev_vis_ts_ms = state->ts_ms;
    g_vi_fuse.prev_vis_valid = true;

    state->deb.vio_fused = true;
    return true;
}

static void chooseCurrentKeyframeSeed(bool prefer_visual_metric_pose,
                                      quat* q_wc,
                                      vec3* p_w,
                                      quat* q_wc_imu,
                                      vec3* p_w_imu) {
    if (q_wc == nullptr || p_w == nullptr || q_wc_imu == nullptr || p_w_imu == nullptr) return;

    if (prefer_visual_metric_pose && g_vis_q.coeffs().allFinite() && g_vis_p.allFinite()) {
        *q_wc = normalizeQ(g_vis_q);
        *p_w = g_vis_p;
    } else if (g_curr_imu_cam_pose_valid) {
        *q_wc = normalizeQ(g_curr_imu_cam_q);
        *p_w = g_curr_imu_cam_p;
    } else {
        *q_wc = normalizeQ(g_vis_q);
        *p_w = g_vis_p;
    }

    if (g_curr_imu_cam_pose_valid) {
        *q_wc_imu = normalizeQ(g_curr_imu_cam_q);
        *p_w_imu = g_curr_imu_cam_p;
    } else {
        *q_wc_imu = *q_wc;
        *p_w_imu = *p_w;
    }
}

static bool refinePosePoseOnly(const TrackerOutput& track,
                               const quat& q_init_wc,
                               const vec3& p_init_w,
                               quat* q_refined_wc,
                               vec3* p_refined_w,
                               int* n_corr,
                               int* n_inliers,
                               double* mean_reproj_px) {
    if (q_refined_wc == nullptr || p_refined_w == nullptr || n_corr == nullptr ||
        n_inliers == nullptr || mean_reproj_px == nullptr) {
        return false;
    }

    *q_refined_wc = q_init_wc;
    *p_refined_w = p_init_w;
    *n_corr = 0;
    *n_inliers = 0;
    *mean_reproj_px = 0.0;

    if (!g_vio_cfg.vio.pose_refine_enable || g_vio_cfg.cam.K.empty()) return false;

    std::vector<cv::Point3f> pts3;
    std::vector<cv::Point2f> pts2;
    std::vector<int> ids;
    buildPoseCorrespondencesUndistPx(track, &pts3, &pts2, &ids);
    *n_corr = static_cast<int>(pts3.size());
    if (*n_corr < std::max(6, g_vio_cfg.vio.pose_refine_min_points)) return false;

    const double fx = g_vio_cfg.cam.K.at<double>(0, 0);
    const double fy = g_vio_cfg.cam.K.at<double>(1, 1);
    const double cx = g_vio_cfg.cam.K.at<double>(0, 2);
    const double cy = g_vio_cfg.cam.K.at<double>(1, 2);
    const double huber_px = std::max(0.1, g_vio_cfg.vio.pose_refine_huber_px);
    const double inlier_px = std::max(0.1, g_vio_cfg.vio.pose_refine_reproj_px);
    const double stop_dx = std::max(1e-9, g_vio_cfg.vio.pose_refine_stop_dx);

    mat3 R_cw = q_init_wc.toRotationMatrix().transpose();
    vec3 t_cw = -R_cw * p_init_w;

    for (int iter = 0; iter < std::max(1, g_vio_cfg.vio.pose_refine_max_iters); ++iter) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
        int used = 0;

        for (size_t i = 0; i < pts3.size() && i < pts2.size(); ++i) {
            const vec3 p_w(pts3[i].x, pts3[i].y, pts3[i].z);
            const vec3 p_c = R_cw * p_w + t_cw;
            if (!p_c.allFinite() || p_c.z() <= 1e-9) continue;

            const double inv_z = 1.0 / p_c.z();
            const double inv_z2 = inv_z * inv_z;
            const double u = fx * p_c.x() * inv_z + cx;
            const double v = fy * p_c.y() * inv_z + cy;
            const Eigen::Vector2d r(
                static_cast<double>(pts2[i].x) - u,
                static_cast<double>(pts2[i].y) - v);

            const double err = r.norm();
            if (!std::isfinite(err) || err > 4.0 * inlier_px) continue;

            Eigen::Matrix<double, 2, 3> Jproj;
            Jproj <<
                fx * inv_z, 0.0, -fx * p_c.x() * inv_z2,
                0.0, fy * inv_z, -fy * p_c.y() * inv_z2;

            Eigen::Matrix<double, 2, 6> J;
            J.leftCols<3>() = Jproj * skewMat(p_c);
            J.rightCols<3>() = -Jproj;

            const double w = (err <= huber_px) ? 1.0 : (huber_px / err);
            H.noalias() += w * (J.transpose() * J);
            b.noalias() += -w * (J.transpose() * r);
            ++used;
        }

        if (used < std::max(6, g_vio_cfg.vio.pose_refine_min_points / 2)) {
            return false;
        }

        H += 1e-6 * Eigen::Matrix<double, 6, 6>::Identity();
        const Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);
        if (!dx.allFinite()) return false;

        const mat3 dR = expSO3Std(dx.head<3>());
        R_cw = dR * R_cw;
        t_cw = dR * t_cw + dx.tail<3>();

        if (dx.norm() < stop_dx) break;
    }

    const mat3 R_wc = R_cw.transpose();
    *q_refined_wc = normalizeQ(quat(R_wc));
    *p_refined_w = -R_wc * t_cw;

    double sum_err = 0.0;
    int good = 0;
    for (size_t i = 0; i < pts3.size() && i < pts2.size(); ++i) {
        const vec3 p_w(pts3[i].x, pts3[i].y, pts3[i].z);
        cv::Point2f px_proj;
        if (!projectWorldPointPx(p_w, *q_refined_wc, *p_refined_w, &px_proj)) continue;
        const double err = cv::norm(px_proj - pts2[i]);
        if (!std::isfinite(err) || err > inlier_px) continue;
        sum_err += err;
        ++good;
    }

    *n_inliers = good;
    *mean_reproj_px = (good > 0) ? (sum_err / good) : 0.0;
    return good >= std::max(6, g_vio_cfg.vio.pose_refine_min_points);
}

static bool runLocalWindowBa(const TrackerOutput& track,
                             const quat& q_init_wc,
                             const vec3& p_init_w,
                             quat* q_ba_wc,
                             vec3* p_ba_w,
                             int* n_landmarks,
                             int* n_obs,
                             int* n_inliers,
                             double* mean_reproj_px) {
    if (q_ba_wc == nullptr || p_ba_w == nullptr || n_landmarks == nullptr ||
        n_obs == nullptr || n_inliers == nullptr || mean_reproj_px == nullptr) {
        return false;
    }

    *q_ba_wc = q_init_wc;
    *p_ba_w = p_init_w;
    *n_landmarks = 0;
    *n_obs = 0;
    *n_inliers = 0;
    *mean_reproj_px = 0.0;

    if (!g_vio_cfg.vio.local_ba_enable || g_vio_cfg.cam.K.empty()) return false;

    std::vector<LocalBaLandmark> lms = collectLocalBaLandmarks(track);
    *n_landmarks = static_cast<int>(lms.size());
    if (*n_landmarks < std::max(4, g_vio_cfg.vio.local_ba_min_landmarks)) return false;

    const double fx = g_vio_cfg.cam.K.at<double>(0, 0);
    const double fy = g_vio_cfg.cam.K.at<double>(1, 1);
    const double cx = g_vio_cfg.cam.K.at<double>(0, 2);
    const double cy = g_vio_cfg.cam.K.at<double>(1, 2);
    const double huber_px = std::max(0.1, g_vio_cfg.vio.local_ba_huber_px);
    const double inlier_px = std::max(0.1, g_vio_cfg.vio.local_ba_reproj_px);
    const double stop_dx = std::max(1e-9, g_vio_cfg.vio.local_ba_stop_dx);
    const double min_improve_px = std::max(0.0, g_vio_cfg.vio.local_ba_min_improve_px);
    const double max_rot_delta_deg = std::max(0.0, g_vio_cfg.vio.local_ba_accept_rot_deg);
    const double max_pos_delta_m = std::max(0.0, g_vio_cfg.vio.local_ba_accept_pos_m);

    mat3 R_cw = q_init_wc.toRotationMatrix().transpose();
    vec3 t_cw = -R_cw * p_init_w;

    Eigen::VectorXd rho = Eigen::VectorXd::Zero(lms.size());
    for (int i = 0; i < static_cast<int>(lms.size()); ++i) {
        rho(i) = std::max(1e-6, lms[i].lm->rho);
    }

    auto evaluateCurrentFrame =
        [&](const quat& q_wc_eval,
            const vec3& p_w_eval,
            const Eigen::VectorXd& rho_eval,
            std::vector<vec3>* p_w_cache,
            std::vector<double>* err_cache,
            int* ninliers,
            double* mean_err_px) -> bool {
            if (ninliers == nullptr || mean_err_px == nullptr) return false;
            *ninliers = 0;
            *mean_err_px = 0.0;
            if (p_w_cache != nullptr) p_w_cache->assign(lms.size(), vec3::Zero());
            if (err_cache != nullptr) err_cache->assign(lms.size(), std::numeric_limits<double>::infinity());

            double sum_err = 0.0;
            int good = 0;
            for (int i = 0; i < static_cast<int>(lms.size()); ++i) {
                const LandmarkState& lm = *lms[i].lm;
                const KeyframeState* anchor_kf = findKeyframeById(lm.anchor_image_id);
                if (anchor_kf == nullptr) continue;

                const double rho_i = std::max(1e-6, rho_eval(i));
                const vec3 p_anchor(
                    static_cast<double>(lm.anchor_un.x) / rho_i,
                    static_cast<double>(lm.anchor_un.y) / rho_i,
                    1.0 / rho_i);
                const vec3 p_w = anchor_kf->q_wc.toRotationMatrix() * p_anchor + anchor_kf->p_w;
                if (!p_w.allFinite()) continue;

                cv::Point2f px_proj;
                if (!projectWorldPointPx(p_w, q_wc_eval, p_w_eval, &px_proj)) continue;

                const double err = cv::norm(px_proj - lms[i].current_px);
                if (!std::isfinite(err)) continue;

                if (p_w_cache != nullptr) {
                    (*p_w_cache)[i] = p_w;
                }
                if (err_cache != nullptr) {
                    (*err_cache)[i] = err;
                }

                if (err <= inlier_px) {
                    ++(*ninliers);
                }
                sum_err += err;
                ++good;
            }

            if (good <= 0) return false;
            *mean_err_px = sum_err / good;
            return true;
        };

    int init_inliers = 0;
    double init_mean_reproj_px = 0.0;
    if (!evaluateCurrentFrame(q_init_wc, p_init_w, rho, nullptr, nullptr, &init_inliers, &init_mean_reproj_px)) {
        return false;
    }

    for (int iter = 0; iter < std::max(1, g_vio_cfg.vio.local_ba_max_iters); ++iter) {
        const int dim = 6 + static_cast<int>(lms.size());
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(dim);
        int used_obs = 0;

        for (int i = 0; i < static_cast<int>(lms.size()); ++i) {
            const LandmarkState& lm = *lms[i].lm;
            const KeyframeState* anchor_kf = findKeyframeById(lm.anchor_image_id);
            if (anchor_kf == nullptr) continue;

            const double rho_i = std::max(1e-6, rho(i));
            const vec3 p_anchor(
                static_cast<double>(lm.anchor_un.x) / rho_i,
                static_cast<double>(lm.anchor_un.y) / rho_i,
                1.0 / rho_i);
            const vec3 dp_anchor_drho(
                -static_cast<double>(lm.anchor_un.x) / (rho_i * rho_i),
                -static_cast<double>(lm.anchor_un.y) / (rho_i * rho_i),
                -1.0 / (rho_i * rho_i));
            const mat3 R_anchor = anchor_kf->q_wc.toRotationMatrix();
            const vec3 p_w = R_anchor * p_anchor + anchor_kf->p_w;
            const vec3 dpw_drho = R_anchor * dp_anchor_drho;

            auto accumulateObs = [&](const mat3& R_cw_obs,
                                     const vec3& t_cw_obs,
                                     const cv::Point2f& px_obs,
                                     bool current_pose) {
                const vec3 p_c = R_cw_obs * p_w + t_cw_obs;
                if (!p_c.allFinite() || p_c.z() <= 1e-9) return;

                const double inv_z = 1.0 / p_c.z();
                const double inv_z2 = inv_z * inv_z;
                const double u = fx * p_c.x() * inv_z + cx;
                const double v = fy * p_c.y() * inv_z + cy;
                const Eigen::Vector2d r(
                    static_cast<double>(px_obs.x) - u,
                    static_cast<double>(px_obs.y) - v);

                const double err = r.norm();
                if (!std::isfinite(err) || err > 4.0 * inlier_px) return;

                Eigen::Matrix<double, 2, 3> Jproj;
                Jproj <<
                    fx * inv_z, 0.0, -fx * p_c.x() * inv_z2,
                    0.0, fy * inv_z, -fy * p_c.y() * inv_z2;

                const Eigen::Vector2d J_rho = -Jproj * (R_cw_obs * dpw_drho);
                const double w = (err <= huber_px) ? 1.0 : (huber_px / err);

                const int rho_idx = 6 + i;
                H(rho_idx, rho_idx) += w * J_rho.dot(J_rho);
                b(rho_idx) += -w * J_rho.dot(r);

                if (current_pose) {
                    Eigen::Matrix<double, 2, 6> J_pose;
                    J_pose.leftCols<3>() = Jproj * skewMat(p_c);
                    J_pose.rightCols<3>() = -Jproj;

                    H.topLeftCorner<6, 6>().noalias() += w * (J_pose.transpose() * J_pose);
                    H.block(0, rho_idx, 6, 1).noalias() += w * (J_pose.transpose() * J_rho);
                    H.block(rho_idx, 0, 1, 6) = H.block(0, rho_idx, 6, 1).transpose();
                    b.head<6>().noalias() += -w * (J_pose.transpose() * r);
                }

                ++used_obs;
            };

            for (const LocalBaObs& obs : lms[i].fixed_obs) {
                mat3 R_cw_obs = mat3::Identity();
                vec3 t_cw_obs = vec3::Zero();
                worldPoseToExtrinsics(obs.kf->q_wc, obs.kf->p_w, &R_cw_obs, &t_cw_obs);
                accumulateObs(R_cw_obs, t_cw_obs, obs.px, false);
            }

            accumulateObs(R_cw, t_cw, lms[i].current_px, true);
        }

        *n_obs = used_obs;
        if (used_obs < std::max(8, g_vio_cfg.vio.local_ba_min_landmarks)) return false;

        H += 1e-6 * Eigen::MatrixXd::Identity(dim, dim);
        const Eigen::VectorXd dx = H.ldlt().solve(b);
        if (!dx.allFinite()) return false;

        const mat3 dR = expSO3Std(dx.head<3>());
        R_cw = dR * R_cw;
        t_cw = dR * t_cw + dx.segment<3>(3);
        for (int i = 0; i < static_cast<int>(lms.size()); ++i) {
            rho(i) = std::max(1e-6, rho(i) + dx(6 + i));
        }

        if (dx.norm() < stop_dx) break;
    }

    const mat3 R_wc = R_cw.transpose();
    *q_ba_wc = normalizeQ(quat(R_wc));
    *p_ba_w = -R_wc * t_cw;

    std::vector<vec3> p_w_cache;
    std::vector<double> err_cache;
    if (!evaluateCurrentFrame(*q_ba_wc, *p_ba_w, rho, &p_w_cache, &err_cache, n_inliers, mean_reproj_px)) {
        return false;
    }

    const double rot_delta_deg = quatDeltaDeg(q_init_wc, *q_ba_wc);
    const double pos_delta_m = (*p_ba_w - p_init_w).norm();
    const bool enough_inliers = *n_inliers >= std::max(6, g_vio_cfg.vio.local_ba_min_landmarks / 2);
    const bool improved =
        (*mean_reproj_px <= init_mean_reproj_px - min_improve_px) ||
        (*mean_reproj_px <= 0.9 * init_mean_reproj_px);
    const bool bounded_pose =
        rot_delta_deg <= max_rot_delta_deg &&
        pos_delta_m <= max_pos_delta_m;

    if (!enough_inliers || !improved || !bounded_pose) {
        *q_ba_wc = q_init_wc;
        *p_ba_w = p_init_w;
        *n_inliers = 0;
        *mean_reproj_px = 0.0;
        return false;
    }

    for (int i = 0; i < static_cast<int>(lms.size()); ++i) {
        if (i >= static_cast<int>(p_w_cache.size()) || i >= static_cast<int>(err_cache.size())) continue;
        LandmarkState& lm = *lms[i].lm;
        if (!p_w_cache[i].allFinite() || !std::isfinite(err_cache[i])) continue;
        lm.rho = std::max(1e-6, rho(i));
        lm.p_w = p_w_cache[i];
        lm.last_reproj_px = err_cache[i];
    }

    return true;
}

static bool runCloneFactorUpdate(const TrackerOutput& track,
                                 const quat& q_init_wc,
                                 const vec3& p_init_w,
                                 quat* q_clone_wc,
                                 vec3* p_clone_w,
                                 int* n_features,
                                 int* n_obs,
                                 int* n_inliers,
                                 double* mean_reproj_px) {
    if (q_clone_wc == nullptr || p_clone_w == nullptr || n_features == nullptr ||
        n_obs == nullptr || n_inliers == nullptr || mean_reproj_px == nullptr) {
        return false;
    }

    *q_clone_wc = q_init_wc;
    *p_clone_w = p_init_w;
    *n_features = 0;
    *n_obs = 0;
    *n_inliers = 0;
    *mean_reproj_px = 0.0;

    if (!g_vio_cfg.vio.clone_factor_enable || g_vio_cfg.cam.K.empty()) return false;

    std::vector<LocalBaLandmark> lms = collectLocalBaLandmarks(track);
    *n_features = static_cast<int>(lms.size());
    if (*n_features < std::max(4, g_vio_cfg.vio.local_ba_min_landmarks)) return false;

    const double fx = g_vio_cfg.cam.K.at<double>(0, 0);
    const double fy = g_vio_cfg.cam.K.at<double>(1, 1);
    const double cx = g_vio_cfg.cam.K.at<double>(0, 2);
    const double cy = g_vio_cfg.cam.K.at<double>(1, 2);
    const double huber_px = std::max(0.1, g_vio_cfg.vio.local_ba_huber_px);
    const double inlier_px = std::max(0.1, g_vio_cfg.vio.local_ba_reproj_px);
    const double stop_dx = std::max(1e-9, g_vio_cfg.vio.local_ba_stop_dx);
    const double min_improve_px = std::max(0.0, g_vio_cfg.vio.local_ba_min_improve_px);
    const double max_rot_delta_deg = std::max(0.0, g_vio_cfg.vio.local_ba_accept_rot_deg);
    const double max_pos_delta_m = std::max(0.0, g_vio_cfg.vio.local_ba_accept_pos_m);

    mat3 R_cw = q_init_wc.toRotationMatrix().transpose();
    vec3 t_cw = -R_cw * p_init_w;

    Eigen::VectorXd rho = Eigen::VectorXd::Zero(lms.size());
    for (int i = 0; i < static_cast<int>(lms.size()); ++i) {
        rho(i) = std::max(1e-6, lms[i].lm->rho);
    }

    auto evaluateCurrentFrame =
        [&](const quat& q_wc_eval,
            const vec3& p_w_eval,
            const Eigen::VectorXd& rho_eval,
            std::vector<vec3>* p_w_cache,
            std::vector<double>* err_cache,
            int* ninliers,
            double* mean_err_px) -> bool {
            if (ninliers == nullptr || mean_err_px == nullptr) return false;
            *ninliers = 0;
            *mean_err_px = 0.0;
            if (p_w_cache != nullptr) p_w_cache->assign(lms.size(), vec3::Zero());
            if (err_cache != nullptr) err_cache->assign(lms.size(), std::numeric_limits<double>::infinity());

            double sum_err = 0.0;
            int good = 0;
            for (int i = 0; i < static_cast<int>(lms.size()); ++i) {
                const LandmarkState& lm = *lms[i].lm;
                const KeyframeState* anchor_kf = findKeyframeById(lm.anchor_image_id);
                if (anchor_kf == nullptr) continue;

                const double rho_i = std::max(1e-6, rho_eval(i));
                const vec3 p_anchor(
                    static_cast<double>(lm.anchor_un.x) / rho_i,
                    static_cast<double>(lm.anchor_un.y) / rho_i,
                    1.0 / rho_i);
                const vec3 p_w = anchor_kf->q_wc.toRotationMatrix() * p_anchor + anchor_kf->p_w;
                if (!p_w.allFinite()) continue;

                cv::Point2f px_proj;
                if (!projectWorldPointPx(p_w, q_wc_eval, p_w_eval, &px_proj)) continue;

                const double err = cv::norm(px_proj - lms[i].current_px);
                if (!std::isfinite(err)) continue;

                if (p_w_cache != nullptr) {
                    (*p_w_cache)[i] = p_w;
                }
                if (err_cache != nullptr) {
                    (*err_cache)[i] = err;
                }
                if (err <= inlier_px) {
                    ++(*ninliers);
                }
                sum_err += err;
                ++good;
            }

            if (good <= 0) return false;
            *mean_err_px = sum_err / good;
            return true;
        };

    int init_inliers = 0;
    double init_mean_reproj_px = 0.0;
    if (!evaluateCurrentFrame(q_init_wc, p_init_w, rho, nullptr, nullptr, &init_inliers, &init_mean_reproj_px)) {
        return false;
    }

    for (int iter = 0; iter < std::max(1, g_vio_cfg.vio.local_ba_max_iters); ++iter) {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
        int used_obs = 0;

        struct FeatureLinearization {
            bool valid = false;
            Eigen::Matrix<double, 6, 1> Atb = Eigen::Matrix<double, 6, 1>::Zero();
            double s = 0.0;
            double br = 0.0;
        };
        std::vector<FeatureLinearization> feat_lin(lms.size());

        for (int i = 0; i < static_cast<int>(lms.size()); ++i) {
            const LandmarkState& lm = *lms[i].lm;
            const KeyframeState* anchor_kf = findKeyframeById(lm.anchor_image_id);
            if (anchor_kf == nullptr) continue;

            const double rho_i = std::max(1e-6, rho(i));
            const vec3 p_anchor(
                static_cast<double>(lm.anchor_un.x) / rho_i,
                static_cast<double>(lm.anchor_un.y) / rho_i,
                1.0 / rho_i);
            const vec3 dp_anchor_drho(
                -static_cast<double>(lm.anchor_un.x) / (rho_i * rho_i),
                -static_cast<double>(lm.anchor_un.y) / (rho_i * rho_i),
                -1.0 / (rho_i * rho_i));
            const mat3 R_anchor = anchor_kf->q_wc.toRotationMatrix();
            const vec3 p_w = R_anchor * p_anchor + anchor_kf->p_w;
            const vec3 dpw_drho = R_anchor * dp_anchor_drho;
            if (!p_w.allFinite() || !dpw_drho.allFinite()) continue;

            Eigen::Matrix<double, 6, 6> AtA = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> Atb = Eigen::Matrix<double, 6, 1>::Zero();
            Eigen::Matrix<double, 6, 1> Atr = Eigen::Matrix<double, 6, 1>::Zero();
            double s = 0.0;
            double br = 0.0;
            int feat_obs = 0;

            auto accumulateObs =
                [&](const mat3& R_cw_obs,
                    const vec3& t_cw_obs,
                    const cv::Point2f& px_obs,
                    bool current_pose) {
                    const vec3 p_c = R_cw_obs * p_w + t_cw_obs;
                    if (!p_c.allFinite() || p_c.z() <= 1e-9) return;

                    const double inv_z = 1.0 / p_c.z();
                    const double inv_z2 = inv_z * inv_z;
                    const double u = fx * p_c.x() * inv_z + cx;
                    const double v = fy * p_c.y() * inv_z + cy;
                    const Eigen::Vector2d r(
                        static_cast<double>(px_obs.x) - u,
                        static_cast<double>(px_obs.y) - v);

                    const double err = r.norm();
                    if (!std::isfinite(err) || err > 4.0 * inlier_px) return;

                    Eigen::Matrix<double, 2, 3> Jproj;
                    Jproj <<
                        fx * inv_z, 0.0, -fx * p_c.x() * inv_z2,
                        0.0, fy * inv_z, -fy * p_c.y() * inv_z2;

                    const Eigen::Vector2d Jr = -Jproj * (R_cw_obs * dpw_drho);
                    const double w = (err <= huber_px) ? 1.0 : (huber_px / err);

                    s += w * Jr.dot(Jr);
                    br += w * Jr.dot(r);

                    if (current_pose) {
                        Eigen::Matrix<double, 2, 6> Jx;
                        Jx.leftCols<3>() = Jproj * skewMat(p_c);
                        Jx.rightCols<3>() = -Jproj;
                        AtA.noalias() += w * (Jx.transpose() * Jx);
                        Atb.noalias() += w * (Jx.transpose() * Jr);
                        Atr.noalias() += w * (Jx.transpose() * r);
                    }

                    ++feat_obs;
                    ++used_obs;
                };

            for (const LocalBaObs& obs : lms[i].fixed_obs) {
                mat3 R_cw_obs = mat3::Identity();
                vec3 t_cw_obs = vec3::Zero();
                worldPoseToExtrinsics(obs.kf->q_wc, obs.kf->p_w, &R_cw_obs, &t_cw_obs);
                accumulateObs(R_cw_obs, t_cw_obs, obs.px, false);
            }

            accumulateObs(R_cw, t_cw, lms[i].current_px, true);

            if (feat_obs < 2 || s <= 1e-12) continue;

            H.noalias() += AtA - (Atb * Atb.transpose()) / s;
            g.noalias() += -Atr + Atb * (br / s);
            feat_lin[i].valid = true;
            feat_lin[i].Atb = Atb;
            feat_lin[i].s = s;
            feat_lin[i].br = br;
        }

        *n_obs = used_obs;
        if (used_obs < std::max(8, g_vio_cfg.vio.local_ba_min_landmarks)) return false;

        H += 1e-6 * Eigen::Matrix<double, 6, 6>::Identity();
        const Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(g);
        if (!dx.allFinite()) return false;

        const mat3 dR = expSO3Std(dx.head<3>());
        R_cw = dR * R_cw;
        t_cw = dR * t_cw + dx.tail<3>();

        for (int i = 0; i < static_cast<int>(lms.size()); ++i) {
            if (!feat_lin[i].valid || feat_lin[i].s <= 1e-12) continue;
            const double drho = (feat_lin[i].br - feat_lin[i].Atb.dot(dx)) / feat_lin[i].s;
            rho(i) = std::max(1e-6, rho(i) + drho);
        }

        if (dx.norm() < stop_dx) break;
    }

    const mat3 R_wc = R_cw.transpose();
    *q_clone_wc = normalizeQ(quat(R_wc));
    *p_clone_w = -R_wc * t_cw;

    std::vector<vec3> p_w_cache;
    std::vector<double> err_cache;
    if (!evaluateCurrentFrame(*q_clone_wc, *p_clone_w, rho, &p_w_cache, &err_cache, n_inliers, mean_reproj_px)) {
        return false;
    }

    const double rot_delta_deg = quatDeltaDeg(q_init_wc, *q_clone_wc);
    const double pos_delta_m = (*p_clone_w - p_init_w).norm();
    const bool enough_inliers = *n_inliers >= std::max(6, g_vio_cfg.vio.local_ba_min_landmarks / 2);
    const bool improved =
        (*mean_reproj_px <= init_mean_reproj_px - min_improve_px) ||
        (*mean_reproj_px <= 0.9 * init_mean_reproj_px);
    const bool bounded_pose =
        rot_delta_deg <= max_rot_delta_deg &&
        pos_delta_m <= max_pos_delta_m;

    if (!enough_inliers || !improved || !bounded_pose) {
        *q_clone_wc = q_init_wc;
        *p_clone_w = p_init_w;
        *n_inliers = 0;
        *mean_reproj_px = 0.0;
        return false;
    }

    for (int i = 0; i < static_cast<int>(lms.size()); ++i) {
        if (i >= static_cast<int>(p_w_cache.size()) || i >= static_cast<int>(err_cache.size())) continue;
        LandmarkState& lm = *lms[i].lm;
        if (!p_w_cache[i].allFinite() || !std::isfinite(err_cache[i])) continue;
        lm.rho = std::max(1e-6, rho(i));
        lm.p_w = p_w_cache[i];
        lm.last_reproj_px = err_cache[i];
    }

    return true;
}

static bool estimatePosePnP(const TrackerOutput& track,
                            quat* q_wc,
                            vec3* p_w,
                            int* n_corr,
                            int* n_inliers,
                            double* mean_reproj_px) {
    if (q_wc == nullptr || p_w == nullptr || n_corr == nullptr || n_inliers == nullptr || mean_reproj_px == nullptr) {
        return false;
    }

    *q_wc = quat::Identity();
    *p_w = vec3::Zero();
    *n_corr = 0;
    *n_inliers = 0;
    *mean_reproj_px = 0.0;

    if (g_vio_cfg.cam.K.empty()) return false;

    std::vector<cv::Point3f> pts3;
    std::vector<cv::Point2f> pts2;
    std::vector<int> ids;
    buildPnPCorrespondences(track, &pts3, &pts2, &ids);
    *n_corr = static_cast<int>(pts3.size());
    if (*n_corr < std::max(4, g_vio_cfg.vio.pnp_min_points)) return false;

    cv::Mat rvec, tvec, inliers;
    const bool ok = cv::solvePnPRansac(
        pts3,
        pts2,
        g_vio_cfg.cam.K,
        g_vio_cfg.cam.D,
        rvec,
        tvec,
        false,
        200,
        std::max(0.1, g_vio_cfg.vio.pnp_reproj_px),
        std::clamp(g_vio_cfg.vio.pnp_conf, 0.5, 0.9999),
        inliers,
        cv::SOLVEPNP_ITERATIVE);
    if (!ok || inliers.empty()) return false;

    *n_inliers = inliers.rows;
    if (*n_inliers < std::max(4, g_vio_cfg.vio.pnp_min_points)) return false;

    cv::Mat R_cw_cv;
    cv::Rodrigues(rvec, R_cw_cv);
    R_cw_cv.convertTo(R_cw_cv, CV_64F);
    tvec.convertTo(tvec, CV_64F);

    mat3 R_cw = mat3::Identity();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R_cw(r, c) = R_cw_cv.at<double>(r, c);
        }
    }

    const mat3 R_wc = R_cw.transpose();
    const vec3 t_cw(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0));
    *p_w = -R_wc * t_cw;
    *q_wc = normalizeQ(quat(R_wc));

    std::vector<cv::Point3f> pts3_inliers;
    std::vector<cv::Point2f> pts2_inliers;
    pts3_inliers.reserve(inliers.rows);
    pts2_inliers.reserve(inliers.rows);
    for (int i = 0; i < inliers.rows; ++i) {
        const int idx = inliers.at<int>(i, 0);
        if (idx < 0 || idx >= static_cast<int>(pts3.size())) continue;
        pts3_inliers.push_back(pts3[idx]);
        pts2_inliers.push_back(pts2[idx]);
    }

    std::vector<cv::Point2f> reproj;
    cv::projectPoints(pts3_inliers, rvec, tvec, g_vio_cfg.cam.K, g_vio_cfg.cam.D, reproj);

    double sum_err = 0.0;
    int n_err = 0;
    for (size_t i = 0; i < reproj.size() && i < pts2_inliers.size(); ++i) {
        const double err = cv::norm(reproj[i] - pts2_inliers[i]);
        if (!std::isfinite(err)) continue;
        sum_err += err;
        ++n_err;
    }
    *mean_reproj_px = (n_err > 0) ? (sum_err / n_err) : 0.0;
    return q_wc->coeffs().allFinite() && p_w->allFinite();
}

static int triangulateLandmarks(const KeyframeState& ref_kf,
                                const std::vector<int>& ids,
                                const std::vector<cv::Point2f>& ref_px,
                                const std::vector<cv::Point2f>& ref_un,
                                const std::vector<cv::Point2f>& cur_px,
                                const std::vector<cv::Point2f>& cur_un,
                                const quat& q_wc_cur,
                                const vec3& p_w_cur,
                                int current_image_id,
                                double* mean_reproj_px) {
    if (mean_reproj_px != nullptr) {
        *mean_reproj_px = 0.0;
    }
    (void)ref_px;
    (void)cur_px;

    const int min_pts = std::max(4, g_vio_cfg.vio.triang_min_points);
    if (ids.size() < static_cast<size_t>(min_pts)) return 0;

    const cv::Mat P_ref = projectionFromPose(ref_kf.q_wc, ref_kf.p_w);
    const cv::Mat P_cur = projectionFromPose(q_wc_cur, p_w_cur);

    cv::Mat X4;
    cv::triangulatePoints(P_ref, P_cur, ref_un, cur_un, X4);
    if (X4.empty()) return 0;
    if (X4.type() != CV_64F) {
        X4.convertTo(X4, CV_64F);
    }

    int good = 0;
    double sum_err = 0.0;
    for (int i = 0; i < X4.cols && i < static_cast<int>(ids.size()); ++i) {
        const double w = X4.at<double>(3, i);
        if (!std::isfinite(w) || std::abs(w) < 1e-12) continue;

        const vec3 p_w(
            X4.at<double>(0, i) / w,
            X4.at<double>(1, i) / w,
            X4.at<double>(2, i) / w);
        if (!p_w.allFinite()) continue;

        cv::Point2f reproj_ref;
        cv::Point2f reproj_cur;
        double depth_ref = 0.0;
        double depth_cur = 0.0;
        if (!projectWorldPointPx(p_w, ref_kf.q_wc, ref_kf.p_w, &reproj_ref, &depth_ref)) continue;
        if (!projectWorldPointPx(p_w, q_wc_cur, p_w_cur, &reproj_cur, &depth_cur)) continue;
        if (depth_ref <= 1e-9 || depth_cur <= 1e-9) continue;

        const double err_ref = cv::norm(reproj_ref - normToPixel(ref_un[i]));
        const double err_cur = cv::norm(reproj_cur - normToPixel(cur_un[i]));
        const double err = 0.5 * (err_ref + err_cur);
        if (!std::isfinite(err) || err > g_vio_cfg.vio.triang_max_reproj_px) continue;

        const mat3 R_cw_ref = ref_kf.q_wc.toRotationMatrix().transpose();
        const vec3 p_anchor = R_cw_ref * (p_w - ref_kf.p_w);
        if (!p_anchor.allFinite() || p_anchor.z() <= 1e-9) continue;

        LandmarkState& lm = g_landmarks[ids[i]];
        lm.id = ids[i];
        lm.p_w = p_w;
        lm.obs_count += 1;
        lm.anchor_image_id = ref_kf.image_id;
        lm.anchor_un = cv::Point2f(
            static_cast<float>(p_anchor.x() / p_anchor.z()),
            static_cast<float>(p_anchor.y() / p_anchor.z()));
        lm.rho = 1.0 / p_anchor.z();
        lm.last_image_id = current_image_id;
        lm.last_reproj_px = err;
        ++good;
        sum_err += err;
    }

    if (mean_reproj_px != nullptr && good > 0) {
        *mean_reproj_px = sum_err / good;
    }
    return good;
}

static void pruneLandmarks(int current_image_id) {
    for (auto it = g_landmarks.begin(); it != g_landmarks.end(); ) {
        const LandmarkState& lm = it->second;
        const bool bad_geometry = !lm.p_w.allFinite();
        const bool anchor_outside_window = !keyframeInWindow(lm.anchor_image_id);
        const bool stale = (lm.last_image_id >= 0) && ((current_image_id - lm.last_image_id) > g_vio_cfg.vio.landmark_max_age);
        const bool low_support = lm.obs_count < g_vio_cfg.vio.landmark_min_obs;
        const bool bad_reproj = lm.last_reproj_px > 2.0 * g_vio_cfg.vio.landmark_max_reproj_px;

        if (bad_geometry || anchor_outside_window || bad_reproj || (stale && low_support)) {
            it = g_landmarks.erase(it);
        } else {
            ++it;
        }
    }
}

static void drawVisualOverlay(cv::Mat* frame,
                              const TrackerOutput& track,
                              int n_kf_inliers,
                              int tri_good,
                              int pnp_inliers,
                              int n_landmarks,
                              int kf_age,
                              double parallax_deg,
                              double reproj_px,
                              bool vo_valid) {
    if (frame == nullptr || frame->empty()) return;

    if (frame->channels() == 1) {
        cv::cvtColor(*frame, *frame, cv::COLOR_GRAY2BGR);
    }

    for (size_t i = 0; i < track.tracked_px.size() && i < track.tracked_prev_px.size(); ++i) {
        cv::line(*frame, track.tracked_prev_px[i], track.tracked_px[i], cv::Scalar(0, 180, 0), 1, cv::LINE_AA);
        cv::circle(*frame, track.tracked_px[i], 2, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }

    for (const cv::Point2f& px : track.new_px) {
        cv::circle(*frame, px, 4, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }

    const std::string txt0 = cv::format(
        "trk:%d kf:%d inl:%d pnp:%d tri:%d",
        static_cast<int>(track.tracked_ids.size()),
        kf_age,
        n_kf_inliers,
        pnp_inliers,
        tri_good);
    const std::string txt1 = cv::format(
        "lm:%d para:%.2f repr:%.2f valid:%d",
        n_landmarks,
        parallax_deg,
        reproj_px,
        vo_valid ? 1 : 0);

    cv::putText(*frame, txt0, cv::Point(16, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(30, 30, 30), 3, cv::LINE_AA);
    cv::putText(*frame, txt0, cv::Point(16, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    cv::putText(*frame, txt1, cv::Point(16, 54), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(30, 30, 30), 3, cv::LINE_AA);
    cv::putText(*frame, txt1, cv::Point(16, 54), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}

} // namespace

void vioInit(Config * config) {
    if (config == nullptr) return;

    g_vio_cfg = *config;
    loadImuCameraExtrinsics();
    trackerInit(config);

    g_vio_ready = true;
    g_vis_q = quat::Identity();
    g_vis_p = vec3::Zero();
    g_kf_window.clear();
    g_landmarks.clear();
    g_frames_from_kf = 0;
    g_vi_fuse = ViFusionState{};
    g_inc_vo_ready = false;
    g_inc_prev_imu_p = vec3::Zero();
    g_inc_prev_imu_q = quat::Identity();
    g_inc_prev_ts_ms = 0.0;
    g_curr_imu_cam_pose_valid = false;
    g_curr_imu_cam_q = quat::Identity();
    g_curr_imu_cam_p = vec3::Zero();
}

void vioUpdate(SourceIn * source, StateOut * state) {
    if (!g_vio_ready || source == nullptr || state == nullptr) return;
    if (!g_vio_cfg.gen.color_on) return;
    if (source->frame.empty() || source->frame.cols <= 0 || source->frame.rows <= 0) return;
    if (!std::isfinite(source->frame_tsms)) return;

    if (g_vio_cfg.gen.imu_on) {
        const double ts_err_ms = std::abs(state->ts_ms - source->frame_tsms);
        if (std::isfinite(ts_err_ms) && ts_err_ms > 2.0) {
            Logger(WARN, "[VIO_TS_WARN] state_ts=%.3f frame_ts=%.3f err=%.3f ms. Run imuPreUpdate before vioUpdate.", state->ts_ms, source->frame_tsms, ts_err_ms);
        }
    }

    const vec3 imu_pred_p = state->pos_m;
    const quat imu_pred_q = normalizeQ(state->quat_rad);
    state->deb.imu_xyz = imu_pred_p;
    state->deb.imu_rpy = quatToCameraRpyRad(imu_pred_q);
    updateCurrentImuCameraPose(*state);

    TrackerOutput track;
    if (!trackerTrackFrame(source->frame, &track)) return;
    touchTrackedLandmarks(track);

    int n_inc_corr = 0;
    int n_inc_inliers = 0;
    int n_inc_pose_inliers = 0;
    double inc_parallax_deg = 0.0;
    double inc_step_m = 0.0;
    const bool inc_vo_ok = updateIncrementalVisualOdometry(track, *state, source->frame_tsms, &n_inc_corr, &n_inc_inliers, &n_inc_pose_inliers, &inc_parallax_deg, &inc_step_m);
    if (inc_vo_ok) {
        setVisualDebugFromCameraPose(g_vis_q, g_vis_p, &state->deb);
    }

    double map_err = 0.0;
    if (latestKeyframe() != nullptr && visualMapDiverged(*state, &map_err)) {
        setVisualDebugFromCameraPose(g_vis_q, g_vis_p, &state->deb);
        Logger(WARN, "[VIO_MAP_DIVERGED] img=%d ts=%.3f err=%.3f reset_thr=%.3f vis_p=[%.3f %.3f %.3f] imu_p=[%.3f %.3f %.3f] fused_p=[%.3f %.3f %.3f]",
               track.image_id, source->frame_tsms, map_err, std::max(3.0, 3.0 * visualPoseMaxResidualM(*state)),
               g_vis_p.x(), g_vis_p.y(), g_vis_p.z(), state->deb.imu_xyz.x(), state->deb.imu_xyz.y(), state->deb.imu_xyz.z(), state->pos_m.x(), state->pos_m.y(), state->pos_m.z());
        resetVisualMapToState(track, source->frame_tsms, state, "map_diverged");
        drawVisualOverlay(&source->frame, track, 0, 0, 0, 0, 0, 0.0, 0.0, false);
        return;
    }

    state->deb.vio_valid = false;
    state->deb.vio_rel_valid = false;
    state->deb.vio_trans_valid = false;
    state->deb.vio_pnp_valid = false;
    state->deb.vio_refined = false;
    state->deb.vio_local_ba = false;
    state->deb.vio_clone_factor = false;
    state->deb.vio_fused = false;
    state->deb.vio_matches = 0;
    state->deb.vio_tracked = static_cast<int>(track.tracked_ids.size());
    state->deb.vio_inliers = 0;
    state->deb.vio_pnp_corr = 0;
    state->deb.vio_kf_age = g_frames_from_kf;
    state->deb.vio_window_kfs = static_cast<int>(g_kf_window.size());
    state->deb.vio_ref_age = 0;
    state->deb.vio_kf_parallax_deg = 0.0;
    state->deb.vio_init_tracks = 0;
    state->deb.vio_pose_only_tracks = 0;
    state->deb.vio_explo_tracks = 0;
    state->deb.vio_triangulated = 0;
    state->deb.vio_pnp_inliers = 0;
    state->deb.vio_refine_inliers = 0;
    state->deb.vio_ba_landmarks = 0;
    state->deb.vio_ba_obs = 0;
    state->deb.vio_ba_inliers = 0;
    state->deb.vio_clone_features = 0;
    state->deb.vio_clone_obs = 0;
    state->deb.vio_clone_inliers = 0;
    state->deb.vio_landmarks = static_cast<int>(g_landmarks.size());
    state->deb.vio_reproj_px = 0.0;
    state->deb.vio_refine_reproj_px = 0.0;
    state->deb.vio_ba_reproj_px = 0.0;
    state->deb.vio_clone_reproj_px = 0.0;
    state->deb.vio_fuse_pos_res_m = 0.0;
    state->deb.vio_fuse_vel_res_ms = 0.0;
    state->deb.vio_fuse_ori_res_deg = 0.0;
    state->deb.scale = 0.0f;

    if (latestKeyframe() == nullptr) {
        if (g_curr_imu_cam_pose_valid) {
            g_vis_q = g_curr_imu_cam_q;
            g_vis_p = g_curr_imu_cam_p;
        } else {
            g_vis_q = quat::Identity();
            g_vis_p = vec3::Zero();
        }

        setKeyframeFromTrack(
            track,
            source->frame_tsms,
            g_vis_q,
            g_vis_p,
            g_curr_imu_cam_pose_valid ? &g_curr_imu_cam_q : nullptr,
            g_curr_imu_cam_pose_valid ? &g_curr_imu_cam_p : nullptr);
        g_inc_vo_ready = true;
        g_inc_prev_imu_p = g_curr_imu_cam_pose_valid ? g_curr_imu_cam_p : vec3::Zero();
        g_inc_prev_imu_q = g_curr_imu_cam_pose_valid ? g_curr_imu_cam_q : quat::Identity();
        g_inc_prev_ts_ms = source->frame_tsms;
        setVisualDebugFromCameraPose(g_vis_q, g_vis_p, &state->deb);
        state->deb.vio_window_kfs = static_cast<int>(g_kf_window.size());

        drawVisualOverlay(&source->frame, track, 0, 0, 0, 0, 0, 0.0, 0.0, false);
        const KeyframeState* kf0 = latestKeyframe();
        const int visible = (kf0 != nullptr) ? static_cast<int>(kf0->un_by_id.size()) : 0;
        Logger(INFO, "[VIO_INIT] frame=%d ts=%.3f visible=%d keyframe=1 p=[%.6f %.6f %.6f] rpy=[%.6f %.6f %.6f]", track.image_id, source->frame_tsms, visible, g_vis_p.x(), g_vis_p.y(), g_vis_p.z(), quatToCameraRpyRad(g_vis_q).x(), quatToCameraRpyRad(g_vis_q).y(), quatToCameraRpyRad(g_vis_q).z());
        return;
    }

    const std::unordered_map<int, int> track_len_by_id = buildTrackLengthMap(track);
    RefSelection ref_sel;
    if (!chooseReferenceKeyframe(track, track_len_by_id, &ref_sel)) {
        ++g_frames_from_kf;
        pruneLandmarks(track.image_id);
        setVisualDebugFromCameraPose(g_vis_q, g_vis_p, &state->deb);
        state->deb.vio_rel_valid = inc_vo_ok;
        state->deb.vio_trans_valid = inc_vo_ok && inc_parallax_deg >= g_vio_cfg.vio.min_parallax_deg;
        state->deb.vio_matches = n_inc_corr;
        state->deb.vio_inliers = n_inc_inliers;
        state->deb.vio_kf_parallax_deg = inc_parallax_deg;

        const bool inc_metric_ok = inc_vo_ok && visualPoseConsistent("INC_NO_REF", g_vis_q, g_vis_p, *state, source->frame_tsms, track.image_id);
        state->deb.vio_valid = inc_metric_ok;
        if (inc_metric_ok) {
            runVisualInertialCorrection(state, true, n_inc_inliers, 0.0, inc_parallax_deg);
        }

        bool new_keyframe = false;
        const int n_visible = static_cast<int>(track.tracked_ids.size() + track.new_ids.size());
        if (n_visible >= std::max(5, g_vio_cfg.trk.min_ransac_points)) {
            quat kf_seed_q = quat::Identity();
            vec3 kf_seed_p = vec3::Zero();
            quat kf_seed_q_imu = quat::Identity();
            vec3 kf_seed_p_imu = vec3::Zero();
            chooseCurrentKeyframeSeed(inc_metric_ok, &kf_seed_q, &kf_seed_p, &kf_seed_q_imu, &kf_seed_p_imu);
            setKeyframeFromTrack(
                track,
                source->frame_tsms,
                kf_seed_q,
                kf_seed_p,
                &kf_seed_q_imu,
                &kf_seed_p_imu);
            new_keyframe = true;
        }

        state->deb.vio_window_kfs = static_cast<int>(g_kf_window.size());
        drawVisualOverlay(&source->frame, track, n_inc_inliers, 0, 0, static_cast<int>(g_landmarks.size()), g_frames_from_kf, inc_parallax_deg, 0.0, inc_metric_ok);

        const double vis_imu_dp = (state->deb.vis_xyz - state->deb.imu_xyz).norm();
        Logger(INFO, "[VIO_NO_REF] img=%d ts=%.3f tracked=%zu new=%zu visible=%d inc_ok=%d inc_corr=%d inc_inl=%d inc_pose=%d inc_para=%.3f inc_step=%.4f new_kf=%d win=%d lm=%d vis_p=[%.3f %.3f %.3f] imu_p=[%.3f %.3f %.3f] vis_imu_dp=%.3f",
               track.image_id, source->frame_tsms, track.tracked_ids.size(), track.new_ids.size(), n_visible,
               inc_vo_ok ? 1 : 0, n_inc_corr, n_inc_inliers, n_inc_pose_inliers, inc_parallax_deg, inc_step_m, new_keyframe ? 1 : 0,
               static_cast<int>(g_kf_window.size()), static_cast<int>(g_landmarks.size()),
               state->deb.vis_xyz.x(), state->deb.vis_xyz.y(), state->deb.vis_xyz.z(),
               state->deb.imu_xyz.x(), state->deb.imu_xyz.y(), state->deb.imu_xyz.z(), vis_imu_dp);
        return;
    }

    const KeyframeState& ref_kf = *ref_sel.kf;
    const std::vector<int>& kf_ids = ref_sel.ids;
    const std::vector<cv::Point2f>& kf_px = ref_sel.kf_px;
    const std::vector<cv::Point2f>& kf_un = ref_sel.kf_un;
    const std::vector<cv::Point2f>& cur_px = ref_sel.cur_px;
    const std::vector<cv::Point2f>& cur_un = ref_sel.cur_un;
    const MatchBuckets& buckets = ref_sel.buckets;
    state->deb.vio_init_tracks = static_cast<int>(buckets.init_slam_idx.size());
    state->deb.vio_pose_only_tracks = static_cast<int>(buckets.pose_only_idx.size());
    state->deb.vio_explo_tracks = static_cast<int>(buckets.exploration_idx.size());
    state->deb.vio_window_kfs = static_cast<int>(g_kf_window.size());
    state->deb.vio_ref_age = ref_sel.ref_age;

    mat3 R_kf_curr = mat3::Identity();
    vec3 t_kf_curr_dir = vec3::Zero();
    int n_corr = 0;
    int n_e_inliers = 0;
    int n_pose_inliers = 0;
    double parallax_deg = 0.0;
    const bool rel_ok = estimateRelativeVisualPose(
        kf_un, cur_un, &R_kf_curr, &t_kf_curr_dir,
        &n_corr, &n_e_inliers, &n_pose_inliers, &parallax_deg);
    if (rel_ok && ref_kf.imu_pose_valid && g_curr_imu_cam_pose_valid) {
        parallax_deg = medianParallaxDeg(kf_un, cur_un, cv::Mat(), &ref_kf.q_wc_imu, &g_curr_imu_cam_q);
    }
    const bool trans_ok = rel_ok && parallax_deg >= g_vio_cfg.vio.min_parallax_deg;
    state->deb.vio_rel_valid = rel_ok;
    state->deb.vio_trans_valid = trans_ok;

    state->deb.vio_matches = n_corr;
    state->deb.vio_inliers = n_e_inliers;
    state->deb.vio_kf_age = g_frames_from_kf;
    state->deb.vio_kf_parallax_deg = parallax_deg;

    

    const vec3 prev_p = g_vis_p;

    quat pose_q = g_vis_q;
    vec3 pose_p = g_vis_p;
    bool pose_has_orientation = false;
    bool pose_has_metric = false;
    if (rel_ok) {
        pose_q = normalizeQ(ref_kf.q_wc * quat(R_kf_curr));
        pose_has_orientation = true;
        if (trans_ok) {
            const double step_m = computeStepMetersFromState(ref_kf, *state, source->frame_tsms);
            pose_p = ref_kf.p_w + ref_kf.q_wc.toRotationMatrix() * (step_m * t_kf_curr_dir.normalized());
            pose_has_metric = true;
        }
    }

    int n_pnp_corr = 0;
    int n_pnp_inliers = 0;
    double pnp_reproj_px = 0.0;
    quat pnp_q = quat::Identity();
    vec3 pnp_p = vec3::Zero();
    const bool pnp_raw_ok = estimatePosePnP(track, &pnp_q, &pnp_p, &n_pnp_corr, &n_pnp_inliers, &pnp_reproj_px);
    const bool pnp_ok = pnp_raw_ok && visualPoseConsistent("PNP", pnp_q, pnp_p, *state, source->frame_tsms, track.image_id);
    state->deb.vio_pnp_valid = pnp_ok;
    state->deb.vio_pnp_corr = n_pnp_corr;
    if (pnp_raw_ok && !pnp_ok) {
        Logger(WARN, "[VIO_PNP_IGNORED] img=%d corr=%d inl=%d reproj=%.3f pnp_p=[%.3f %.3f %.3f] imu_p=[%.3f %.3f %.3f]",
               track.image_id, n_pnp_corr, n_pnp_inliers, pnp_reproj_px,
               pnp_p.x(), pnp_p.y(), pnp_p.z(), state->pos_m.x(), state->pos_m.y(), state->pos_m.z());
    }
    const bool use_pnp_pose = pnp_ok && (!pose_has_metric || g_vio_cfg.vio.pnp_use_pose);
    if (use_pnp_pose) {
        pose_q = pnp_q;
        pose_p = pnp_p;
        pose_has_orientation = true;
        pose_has_metric = true;
    }

    int n_refine_corr = 0;
    int n_refine_inliers = 0;
    double refine_reproj_px = 0.0;
    quat refine_q = pose_q;
    vec3 refine_p = pose_p;
    bool refine_ok =
        pose_has_metric &&
        refinePosePoseOnly(track, pose_q, pose_p, &refine_q, &refine_p,
                           &n_refine_corr, &n_refine_inliers, &refine_reproj_px);
    if (refine_ok && !visualPoseConsistent("REFINE", refine_q, refine_p, *state, source->frame_tsms, track.image_id)) {
        refine_ok = false;
    }
    if (refine_ok) {
        pose_q = refine_q;
        pose_p = refine_p;
    }
    state->deb.vio_refined = refine_ok;
    state->deb.vio_refine_inliers = n_refine_inliers;
    state->deb.vio_refine_reproj_px = refine_reproj_px;

    int n_clone_features = 0;
    int n_clone_obs = 0;
    int n_clone_inliers = 0;
    double clone_reproj_px = 0.0;
    quat clone_q = pose_q;
    vec3 clone_p = pose_p;
    bool clone_ok =
        pose_has_metric &&
        runCloneFactorUpdate(track, pose_q, pose_p, &clone_q, &clone_p,
                             &n_clone_features, &n_clone_obs, &n_clone_inliers, &clone_reproj_px);
    if (clone_ok && !visualPoseConsistent("CLONE", clone_q, clone_p, *state, source->frame_tsms, track.image_id)) {
        clone_ok = false;
    }
    if (clone_ok) {
        pose_q = clone_q;
        pose_p = clone_p;
    }
    state->deb.vio_clone_factor = clone_ok;
    state->deb.vio_clone_features = n_clone_features;
    state->deb.vio_clone_obs = n_clone_obs;
    state->deb.vio_clone_inliers = n_clone_inliers;
    state->deb.vio_clone_reproj_px = clone_reproj_px;

    int n_ba_landmarks = 0;
    int n_ba_obs = 0;
    int n_ba_inliers = 0;
    double ba_reproj_px = 0.0;
    quat ba_q = pose_q;
    vec3 ba_p = pose_p;
    bool ba_ok =
        pose_has_metric &&
        runLocalWindowBa(track, pose_q, pose_p, &ba_q, &ba_p,
                         &n_ba_landmarks, &n_ba_obs, &n_ba_inliers, &ba_reproj_px);
    if (ba_ok && !visualPoseConsistent("BA", ba_q, ba_p, *state, source->frame_tsms, track.image_id)) {
        ba_ok = false;
    }
    if (ba_ok) {
        pose_q = ba_q;
        pose_p = ba_p;
    }
    state->deb.vio_local_ba = ba_ok;
    state->deb.vio_ba_landmarks = n_ba_landmarks;
    state->deb.vio_ba_obs = n_ba_obs;
    state->deb.vio_ba_inliers = n_ba_inliers;
    state->deb.vio_ba_reproj_px = ba_reproj_px;

    const bool visual_candidate_available = pose_has_orientation || pnp_ok;
    const quat visual_candidate_q = normalizeQ(pose_q);
    const vec3 visual_candidate_p = pose_p;

    bool pose_valid = pose_has_orientation || pnp_ok;
    const bool final_pose_consistent =
        pose_has_metric ?
        visualPoseConsistent("FINAL", pose_q, pose_p, *state, source->frame_tsms, track.image_id) :
        visualOrientationConsistent("FINAL_ORI", pose_q, *state, source->frame_tsms, track.image_id);
    if (pose_valid && !final_pose_consistent) {
        pose_valid = false;
    }

    std::vector<int> init_ids;
    std::vector<cv::Point2f> init_kf_px, init_kf_un, init_cur_px, init_cur_un;
    selectMatchesByIndex(
        buckets.init_slam_idx,
        kf_ids, kf_px, kf_un, cur_px, cur_un,
        &init_ids, &init_kf_px, &init_kf_un, &init_cur_px, &init_cur_un);

    const bool can_triangulate =
        pose_valid &&
        pose_has_metric &&
        rel_ok &&
        trans_ok &&
        static_cast<int>(init_ids.size()) >= std::max(4, g_vio_cfg.vio.triang_min_points);

    double tri_reproj_px = 0.0;
    int tri_good = 0;
    if (can_triangulate) {
        tri_good = triangulateLandmarks(
            ref_kf, init_ids, init_kf_px, init_kf_un, init_cur_px, init_cur_un,
            pose_q, pose_p, track.image_id, &tri_reproj_px);
    }
    pruneLandmarks(track.image_id);

    if (pose_valid) {
        g_vis_q = normalizeQ(pose_q);
        if (pose_has_metric) {
            g_vis_p = pose_p;
        }
        state->deb.vio_valid = rel_ok || pnp_ok;
        state->deb.scale = pose_has_metric ? static_cast<float>((g_vis_p - prev_p).norm()) : 0.0f;
    }

    if (visual_candidate_available) {
        setVisualDebugFromCameraPose(visual_candidate_q, visual_candidate_p, &state->deb);
    } else {
        setVisualDebugFromCameraPose(g_vis_q, g_vis_p, &state->deb);
    }
    state->deb.vio_pnp_inliers = n_pnp_inliers;
    state->deb.vio_triangulated = tri_good;
    state->deb.vio_landmarks = static_cast<int>(g_landmarks.size());
    state->deb.vio_reproj_px = (tri_good > 0) ? tri_reproj_px : (pnp_ok ? pnp_reproj_px : 0.0);

    const int fuse_inliers =
        ba_ok ? n_ba_inliers :
        (clone_ok ? n_clone_inliers :
        (refine_ok ? n_refine_inliers :
        (pnp_ok ? n_pnp_inliers : n_e_inliers)));
    const double fuse_reproj_px =
        ba_ok ? ba_reproj_px :
        (clone_ok ? clone_reproj_px :
        (refine_ok ? refine_reproj_px : state->deb.vio_reproj_px));
    const bool metric_visual_pose_ok = pose_valid && pose_has_metric;
    runVisualInertialCorrection(state, metric_visual_pose_ok, fuse_inliers, fuse_reproj_px, parallax_deg);

    bool new_keyframe = false;
    if (pose_valid && pose_has_metric) {
        const bool age_trigger = g_frames_from_kf >= std::max(1, g_vio_cfg.vio.keyframe_max_age);
        const bool parallax_trigger = parallax_deg >= g_vio_cfg.vio.keyframe_parallax_deg;
        const bool triang_trigger = tri_good >= std::max(4, g_vio_cfg.vio.triang_min_points / 2);
        const bool enough_init_tracks = static_cast<int>(init_ids.size()) >= std::max(4, g_vio_cfg.vio.triang_min_points / 2);

        if (age_trigger || (parallax_trigger && triang_trigger && enough_init_tracks)) {
            setKeyframeFromTrack(
                track,
                source->frame_tsms,
                g_vis_q,
                g_vis_p,
                g_curr_imu_cam_pose_valid ? &g_curr_imu_cam_q : nullptr,
                g_curr_imu_cam_pose_valid ? &g_curr_imu_cam_p : nullptr);
            new_keyframe = true;
        } else {
            ++g_frames_from_kf;
        }
    } else {
        ++g_frames_from_kf;
        const bool stale_kf = g_frames_from_kf > 2 * std::max(1, g_vio_cfg.vio.keyframe_max_age);
        const int n_visible = static_cast<int>(track.tracked_ids.size() + track.new_ids.size());
        if (stale_kf && n_visible >= std::max(5, g_vio_cfg.trk.min_ransac_points)) {
            quat kf_seed_q = quat::Identity();
            vec3 kf_seed_p = vec3::Zero();
            quat kf_seed_q_imu = quat::Identity();
            vec3 kf_seed_p_imu = vec3::Zero();
            chooseCurrentKeyframeSeed(false, &kf_seed_q, &kf_seed_p, &kf_seed_q_imu, &kf_seed_p_imu);
            setKeyframeFromTrack(
                track,
                source->frame_tsms,
                kf_seed_q,
                kf_seed_p,
                &kf_seed_q_imu,
                &kf_seed_p_imu);
            new_keyframe = true;
        }
    }

    state->deb.vio_window_kfs = static_cast<int>(g_kf_window.size());

    drawVisualOverlay(&source->frame, track, n_e_inliers, tri_good, n_pnp_inliers,
                      static_cast<int>(g_landmarks.size()), ref_sel.ref_age,
                      parallax_deg, state->deb.vio_reproj_px, state->deb.vio_valid);

    const double vis_imu_dp = (state->deb.vis_xyz - state->deb.imu_xyz).norm();
    const double fused_imu_dp = (state->pos_m - state->deb.imu_xyz).norm();
    const double fused_vis_dp = (state->pos_m - state->deb.vis_xyz).norm();

    Logger(INFO, "[VIO_INC] img=%d ok=%d corr=%d inl=%d pose_inl=%d para=%.3f step=%.4f tracked=%zu new=%zu",
           track.image_id, inc_vo_ok ? 1 : 0, n_inc_corr, n_inc_inliers, n_inc_pose_inliers, inc_parallax_deg, inc_step_m, track.tracked_ids.size(), track.new_ids.size());

    Logger(INFO, "[VIO_LAYERS] img=%d ts=%.3f imu_p=[%.3f %.3f %.3f] vis_p=[%.3f %.3f %.3f] fused_p=[%.3f %.3f %.3f] vis_imu_dp=%.3f fused_imu_dp=%.3f fused_vis_dp=%.3f imu_rpy=[%.3f %.3f %.3f] vis_rpy=[%.3f %.3f %.3f] fused_rpy=[%.3f %.3f %.3f] rel=%d trans=%d pnp=%d visual_accepted=%d fuse=%d",
           track.image_id, source->frame_tsms,
           state->deb.imu_xyz.x(), state->deb.imu_xyz.y(), state->deb.imu_xyz.z(),
           state->deb.vis_xyz.x(), state->deb.vis_xyz.y(), state->deb.vis_xyz.z(),
           state->pos_m.x(), state->pos_m.y(), state->pos_m.z(),
           vis_imu_dp, fused_imu_dp, fused_vis_dp,
           state->deb.imu_rpy.x(), state->deb.imu_rpy.y(), state->deb.imu_rpy.z(),
           state->deb.vis_rpy.x(), state->deb.vis_rpy.y(), state->deb.vis_rpy.z(),
           state->rpy_rad.x(), state->rpy_rad.y(), state->rpy_rad.z(),
           rel_ok ? 1 : 0, trans_ok ? 1 : 0, pnp_ok ? 1 : 0, pose_valid ? 1 : 0, state->deb.vio_fused ? 1 : 0);

    Logger(INFO, "[VIO_CHECK] img=%d frame_ts=%.3f state_ts=%.3f imu_p=[%.3f %.3f %.3f] vis_p=[%.3f %.3f %.3f] fused_p=[%.3f %.3f %.3f] vis_imu_dp=%.3f fused_imu_dp=%.3f rel=%d trans=%d pnp=%d fuse=%d para=%.3f repr=%.3f lm=%d",
           track.image_id, source->frame_tsms, state->ts_ms,
           state->deb.imu_xyz.x(), state->deb.imu_xyz.y(), state->deb.imu_xyz.z(),
           state->deb.vis_xyz.x(), state->deb.vis_xyz.y(), state->deb.vis_xyz.z(),
           state->pos_m.x(), state->pos_m.y(), state->pos_m.z(),
           vis_imu_dp, fused_imu_dp,
           rel_ok ? 1 : 0, trans_ok ? 1 : 0, pnp_ok ? 1 : 0, state->deb.vio_fused ? 1 : 0,
           parallax_deg, state->deb.vio_reproj_px, static_cast<int>(g_landmarks.size()));

    Logger(DEBUG,
        "VIS ODO valid=%d rel=%d trans=%d pnp=%d ref=%d cl=%d ba=%d fuse=%d kf_new=%d win=%d refAge=%d init=%d pose=%d explo=%d corr=%d Einl=%d Pinl=%d pnpCorr=%d pnpInl=%d refCorr=%d refInl=%d clFeat=%d clObs=%d clInl=%d baLm=%d baObs=%d baInl=%d tri=%d lm=%d para=%.3fdeg repr=%.3f refRepr=%.3f clRepr=%.3f baRepr=%.3f fusePos=%.3f fuseVel=%.3f fuseOri=%.3f imu_p=[%.6f %.6f %.6f] vis_p=[%.6f %.6f %.6f] fused_p=[%.6f %.6f %.6f] imu_rpy=[%.6f %.6f %.6f] vis_rpy=[%.6f %.6f %.6f] fused_rpy=[%.6f %.6f %.6f]",
        state->deb.vio_valid ? 1 : 0,
        rel_ok ? 1 : 0,
        trans_ok ? 1 : 0,
        pnp_ok ? 1 : 0,
        refine_ok ? 1 : 0,
        clone_ok ? 1 : 0,
        ba_ok ? 1 : 0,
        state->deb.vio_fused ? 1 : 0,
        new_keyframe ? 1 : 0,
        static_cast<int>(g_kf_window.size()),
        ref_sel.ref_age,
        state->deb.vio_init_tracks,
        state->deb.vio_pose_only_tracks,
        state->deb.vio_explo_tracks,
        n_corr,
        n_e_inliers,
        n_pose_inliers,
        n_pnp_corr,
        n_pnp_inliers,
        n_refine_corr,
        n_refine_inliers,
        n_clone_features,
        n_clone_obs,
        n_clone_inliers,
        n_ba_landmarks,
        n_ba_obs,
        n_ba_inliers,
        tri_good,
        static_cast<int>(g_landmarks.size()),
        parallax_deg,
        state->deb.vio_reproj_px,
        state->deb.vio_refine_reproj_px,
        state->deb.vio_clone_reproj_px,
        state->deb.vio_ba_reproj_px,
        state->deb.vio_fuse_pos_res_m,
        state->deb.vio_fuse_vel_res_ms,
        state->deb.vio_fuse_ori_res_deg,
        state->deb.imu_xyz.x(), state->deb.imu_xyz.y(), state->deb.imu_xyz.z(),
        state->deb.vis_xyz.x(), state->deb.vis_xyz.y(), state->deb.vis_xyz.z(),
        state->pos_m.x(), state->pos_m.y(), state->pos_m.z(),
        state->deb.imu_rpy.x(), state->deb.imu_rpy.y(), state->deb.imu_rpy.z(),
        state->deb.vis_rpy.x(), state->deb.vis_rpy.y(), state->deb.vis_rpy.z(),
        state->rpy_rad.x(), state->rpy_rad.y(), state->rpy_rad.z());
    source->frame.release();
    source->imu.clear();
}
