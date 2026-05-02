#pragma once
#include <algorithm>
#include <cctype>
#include <deque>
#include <Eigen/Dense>
#include <iomanip>
#include <limits>
#include <opencv2/opencv.hpp>
#include <sstream>
#include "seconds/logger.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>

typedef Eigen::Vector3d vec3;
typedef Eigen::Vector4d vec4;
typedef Eigen::Quaterniond quat;
typedef Eigen::Matrix<double,16,1> vec16;
typedef Eigen::Matrix3d mat3;
typedef Eigen::Matrix<double, 6, 6> mat6;
typedef Eigen::Matrix<double, 9, 9> mat9;
typedef Eigen::Matrix<double, 9, 6> mat96;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
const double null = std::numeric_limits<double>::quiet_NaN();

enum SourceType{
    SOURCE_RSCAM = 0,
    SOURCE_BAG,
    SOURCE_MP4,
    SOURCE_RTSP,
    SOURCE_CSV,
    SOURCE_PORT,

    SOURCE_COUNT
};

struct imuCal{
    double acc_fps = 0.0;
    double gyr_fps = 0.0;
    double small_angle = 1.745329e-3;
    double batch_ms = 0.0;
    bool stationary_enable = true;
    bool zupt_enable = true;
    bool bias_refine_enable = true;
    int stationary_min_samples = 10;
    double stationary_gyro_mean_max = 2e-2;
    double stationary_acc_mean_tol = 2e-1;
    double stationary_gyro_std_max = 1e-2;
    double stationary_acc_std_max = 8e-2;
    double stationary_bias_alpha = 2e-2;

    vec3 ba = vec3::Zero();
    vec3 bg = vec3::Zero();

    vec3 allanax = vec3::Zero();
    vec3 allanay = vec3::Zero();
    vec3 allanaz = vec3::Zero();

    vec3 allangx = vec3::Zero();
    vec3 allangy = vec3::Zero();
    vec3 allangz = vec3::Zero();

    vec3 gv = vec3(0.0, -9.80665, 0.0);
    cv::Mat T;


    vec3 gyroAllanN() const {
        return vec3(allangx.x(), allangy.x(), allangz.x());
    }

    vec3 gyroAllanK() const {
        return vec3(allangx.z(), allangy.z(), allangz.z());
    }

    vec3 accelAllanN() const {
        return vec3(allanax.x(), allanay.x(), allanaz.x());
    }

    vec3 accelAllanK() const {
        return vec3(allanax.z(), allanay.z(), allanaz.z());
    }

    bool hasValidAllan() const {
        const vec3 Ng = gyroAllanN();
        const vec3 Kg = gyroAllanK();
        const vec3 Na = accelAllanN();
        const vec3 Ka = accelAllanK();
        return (Ng.array() > 0.0).all() && (Kg.array() > 0.0).all() && (Na.array() > 0.0).all() && (Ka.array() > 0.0).all();
    }

};
struct DepOrb{
    float min_dist = 8.0f;
    float quality = 0.01f;
    int lk_max_lvl = 6;
    int max_feat = 2000;
    int min_pts = 80;
    int min_inliers = 40;
    int lk_win_size = 21;
    int patch_r = 1;
};
struct TrackerTune {
    int min_ransac_points = 8;
    double ransac_f_px = 1.5;
    double ransac_conf = 0.99;
    double fb_max_err_px = 0.75;
    int lk_iters = 30;
    double lk_eps = 1e-2;
    int gftt_block_size = 3;
    double gftt_k = 0.04;
};
struct VioTune {
    double visual_step_scale = 1.0;
    int window_size = 8;
    double min_parallax_deg = 0.30;
    double good_parallax_deg = 2.0;
    int min_track_length = 3;
    int keyframe_max_age = 8;
    double keyframe_parallax_deg = 2.0;
    int triang_min_points = 12;
    double triang_max_reproj_px = 2.0;
    int pnp_min_points = 12;
    double pnp_reproj_px = 3.0;
    double pnp_conf = 0.99;
    int landmark_min_obs = 3;
    int landmark_max_age = 30;
    double landmark_max_reproj_px = 2.0;
    bool pnp_use_pose = true;
    bool pose_refine_enable = true;
    int pose_refine_min_points = 20;
    int pose_refine_max_iters = 8;
    double pose_refine_reproj_px = 3.0;
    double pose_refine_huber_px = 2.0;
    double pose_refine_stop_dx = 1e-4;
    bool local_ba_enable = true;
    int local_ba_min_landmarks = 20;
    int local_ba_max_landmarks = 80;
    int local_ba_max_iters = 6;
    double local_ba_reproj_px = 3.0;
    double local_ba_huber_px = 2.0;
    double local_ba_stop_dx = 1e-4;
    double local_ba_min_improve_px = 0.03;
    double local_ba_accept_rot_deg = 0.75;
    double local_ba_accept_pos_m = 0.05;
    bool clone_factor_enable = true;
    bool fuse_enable = true;
    int fuse_min_inliers = 80;
    double fuse_max_reproj_px = 1.5;
    double fuse_min_parallax_deg = 0.8;
    double fuse_pos_gain = 0.30;
    double fuse_vel_gain = 0.15;
    double fuse_ori_gain = 0.05;
    double fuse_max_pos_corr_m = 0.25;
    double fuse_max_vel_corr_ms = 0.80;
    double fuse_max_ori_corr_deg = 1.0;
};
struct General{
    SourceType type;
    std::string input;
    std::string output;

    bool groundt;
    bool color_on;
    bool imu_on;
    bool show;
    bool debug;

    bool plot_tray;             // Plot VIO estimated trayectory
    bool plot_vis_tray;
    bool plot_imu_tray;

    bool plot_gt_with_tray;     // Plot Ground truth with estimated trayectory
    bool plot_gt_with_vis_tray;
    bool plot_gt_with_imu_tray;
    bool plot_2d;

    bool plot_imu;              // Plot Accel and Gyro calibrated and raw 
    bool plot_rpy;              // Plot IMU output as 
    bool plot_vis_rpy;          // Plot visual-only RPY debug
    bool plot_gt_with_rpy;
    bool plot_gt_with_vis_rpy;
};
struct Camera {
    int w;
    int h;
    float fps;

    cv::Mat K;
    cv::Mat D;

    bool is_rgb;
};
struct Config{
    General gen;
    Camera cam;
    DepOrb dorb;
    TrackerTune trk;
    VioTune vio;
    imuCal imu;

public:
    void configSetOnlyImu() {this->gen.color_on = false; this->gen.imu_on = true; this->gen.groundt = false;}
    void configSetOnlyVis() {this->gen.color_on = true; this->gen.imu_on = false; this->gen.groundt = false;}
    void configSetCamFps(float fps) {this->cam.fps = fps;}

    void print(std::ostream& os = std::cout) const {
        os << "\n========================================\n";
        os << "              CONFIG PRINT              \n";
        os << "========================================\n";

        os << "\n[GENERAL]\n";
        os << "  Source type   : " << sourceTypeToString(this->gen.type) << "\n";
        os << "  Input         : " << this->gen.input << "\n";
        os << "  Output        : " << this->gen.output << "\n";
        os << "  Ground truth  : " << boolText(this->gen.groundt) << "\n";
        os << "  Color         : " << boolText(this->gen.color_on) << "\n";
        os << "  IMU           : " << boolText(this->gen.imu_on) << "\n";
        os << "  Show          : " << boolText(this->gen.show) << "\n";
        os << "  Debug         : " << boolText(this->gen.debug) << "\n";

        os << "\n[CAMERA]\n";
        os << "  Width         : " << this->cam.w << "\n";
        os << "  Height        : " << this->cam.h << "\n";
        os << "  FPS           : " << this->cam.fps << "\n";
        os << "  K             :\n" << matToString(this->cam.K, 10) << "\n";
        os << "  D             :\n" << matToString(this->cam.D, 10) << "\n";

        os << "\n[FRONTEND / GT]\n";
        os << "  Min dist      : " << this->dorb.min_dist << "\n";
        os << "  Quality       : " << this->dorb.quality << "\n";
        os << "  LK max lvl    : " << this->dorb.lk_max_lvl << "\n";
        os << "  Max features  : " << this->dorb.max_feat << "\n";
        os << "  Min points    : " << this->dorb.min_pts << "\n";
        os << "  Min inliers   : " << this->dorb.min_inliers << "\n";
        os << "  LK win size   : " << this->dorb.lk_win_size << "\n";
        os << "  Patch radius  : " << this->dorb.patch_r << "\n";

        os << "\n[TRACKER]\n";
        os << "  Min RANSAC pts: " << this->trk.min_ransac_points << "\n";
        os << "  F thresh px   : " << this->trk.ransac_f_px << "\n";
        os << "  RANSAC conf   : " << this->trk.ransac_conf << "\n";
        os << "  FB max err px : " << this->trk.fb_max_err_px << "\n";
        os << "  LK iters      : " << this->trk.lk_iters << "\n";
        os << "  LK eps        : " << this->trk.lk_eps << "\n";
        os << "  GFTT block    : " << this->trk.gftt_block_size << "\n";
        os << "  GFTT k        : " << this->trk.gftt_k << "\n";

        os << "\n[VIO]\n";
        os << "  Vis step scale: " << this->vio.visual_step_scale << "\n";
        os << "  Window size   : " << this->vio.window_size << "\n";
        os << "  Min parallax  : " << this->vio.min_parallax_deg << "\n";
        os << "  Good parallax : " << this->vio.good_parallax_deg << "\n";
        os << "  Min track len : " << this->vio.min_track_length << "\n";
        os << "  KF max age    : " << this->vio.keyframe_max_age << "\n";
        os << "  KF parallax   : " << this->vio.keyframe_parallax_deg << "\n";
        os << "  Tri min pts   : " << this->vio.triang_min_points << "\n";
        os << "  Tri reproj px : " << this->vio.triang_max_reproj_px << "\n";
        os << "  PnP min pts   : " << this->vio.pnp_min_points << "\n";
        os << "  PnP reproj px : " << this->vio.pnp_reproj_px << "\n";
        os << "  PnP conf      : " << this->vio.pnp_conf << "\n";
        os << "  LM min obs    : " << this->vio.landmark_min_obs << "\n";
        os << "  LM max age    : " << this->vio.landmark_max_age << "\n";
        os << "  LM reproj px  : " << this->vio.landmark_max_reproj_px << "\n";
        os << "  PnP use pose  : " << boolText(this->vio.pnp_use_pose) << "\n";
        os << "  Pose refine   : " << boolText(this->vio.pose_refine_enable) << "\n";
        os << "  Ref min pts   : " << this->vio.pose_refine_min_points << "\n";
        os << "  Ref max iters : " << this->vio.pose_refine_max_iters << "\n";
        os << "  Ref reproj px : " << this->vio.pose_refine_reproj_px << "\n";
        os << "  Ref huber px  : " << this->vio.pose_refine_huber_px << "\n";
        os << "  Ref stop dx   : " << this->vio.pose_refine_stop_dx << "\n";
        os << "  Local BA      : " << boolText(this->vio.local_ba_enable) << "\n";
        os << "  BA min lm     : " << this->vio.local_ba_min_landmarks << "\n";
        os << "  BA max lm     : " << this->vio.local_ba_max_landmarks << "\n";
        os << "  BA max iters  : " << this->vio.local_ba_max_iters << "\n";
        os << "  BA reproj px  : " << this->vio.local_ba_reproj_px << "\n";
        os << "  BA huber px   : " << this->vio.local_ba_huber_px << "\n";
        os << "  BA stop dx    : " << this->vio.local_ba_stop_dx << "\n";
        os << "  BA min improve: " << this->vio.local_ba_min_improve_px << "\n";
        os << "  BA max rot deg: " << this->vio.local_ba_accept_rot_deg << "\n";
        os << "  BA max pos m  : " << this->vio.local_ba_accept_pos_m << "\n";
        os << "  Clone factor  : " << boolText(this->vio.clone_factor_enable) << "\n";
        os << "  Fuse enable   : " << boolText(this->vio.fuse_enable) << "\n";
        os << "  Fuse min inl  : " << this->vio.fuse_min_inliers << "\n";
        os << "  Fuse max repr : " << this->vio.fuse_max_reproj_px << "\n";
        os << "  Fuse min para : " << this->vio.fuse_min_parallax_deg << "\n";
        os << "  Fuse pos gain : " << this->vio.fuse_pos_gain << "\n";
        os << "  Fuse vel gain : " << this->vio.fuse_vel_gain << "\n";
        os << "  Fuse ori gain : " << this->vio.fuse_ori_gain << "\n";
        os << "  Fuse max pos  : " << this->vio.fuse_max_pos_corr_m << "\n";
        os << "  Fuse max vel  : " << this->vio.fuse_max_vel_corr_ms << "\n";
        os << "  Fuse max ori  : " << this->vio.fuse_max_ori_corr_deg << "\n";

        os << "\n[PLOTTERS]\n";
        os << "TRAYECTORY PLOTS IN " << std::string(this->gen.plot_2d ? "2D" : "3D") << "\n";
        os << "TRAY             :" << boolText(this->gen.plot_tray) << "\n";
        os << "VIS_TRAY         :" << boolText(this->gen.plot_vis_tray) << "\n";
        os << "IMU_TRAY         :" << boolText(this->gen.plot_imu_tray) << "\n";
        os << "GT_WITH_TRAY     :" << boolText(this->gen.plot_gt_with_tray) << "\n";
        os << "GT_WITH_VIS_TRAY :" << boolText(this->gen.plot_gt_with_vis_tray) << "\n";
        os << "GT_WITH_IMU_TRAY :" << boolText(this->gen.plot_gt_with_imu_tray) << "\n";

        os << "IMU              :" << boolText(this->gen.plot_imu) << "\n";
        os << "RPY              :" << boolText(this->gen.plot_rpy) << "\n";
        os << "VIS_RPY          :" << boolText(this->gen.plot_vis_rpy) << "\n";
        os << "GT_WITH_RPY      :" << boolText(this->gen.plot_gt_with_rpy) << "\n";
        os << "GT_WITH_VIS_RPY  :" << boolText(this->gen.plot_gt_with_vis_rpy) << "\n";

        os << "\n[IMU CALIBRATION]\n";
        os << "  ACC FPS       : " << this->imu.acc_fps << "\n";
        os << "  GYR FPS       : " << this->imu.gyr_fps << "\n";
        os << "  BATCH MS      : " << this->imu.batch_ms << "\n";
        os << "  SMALL ANGLE   : " << this->imu.small_angle << "\n";
        os << "  STATIONARY    : " << boolText(this->imu.stationary_enable) << "\n";
        os << "  ZUPT          : " << boolText(this->imu.zupt_enable) << "\n";
        os << "  BIAS REFINE   : " << boolText(this->imu.bias_refine_enable) << "\n";
        os << "  STAT MIN SAMP : " << this->imu.stationary_min_samples << "\n";
        os << "  STAT GYR MEAN : " << this->imu.stationary_gyro_mean_max << "\n";
        os << "  STAT ACC TOL  : " << this->imu.stationary_acc_mean_tol << "\n";
        os << "  STAT GYR STD  : " << this->imu.stationary_gyro_std_max << "\n";
        os << "  STAT ACC STD  : " << this->imu.stationary_acc_std_max << "\n";
        os << "  STAT BIAS ALP : " << this->imu.stationary_bias_alpha << "\n";
        os << "  ALLAN OK      : " << boolText(this->imu.hasValidAllan()) << "\n";

        os << "\n[RESUMEN DE MODOS]\n";
        os << "  Vision        : " << boolText(this->gen.color_on) << "\n";
        os << "  IMU           : " << boolText(this->gen.imu_on) << "\n";
        os << "  GT            : " << boolText(this->gen.groundt) << "\n";
        os << "  Display       : " << boolText(this->gen.show) << "\n";
        os << "  Debug         : " << boolText(this->gen.debug) << "\n";

        os << "========================================\n";
    }

    bool parseYAML(const char * filename){
        if(toLower(std::string(filename)).find("yaml") == std::string::npos ){
            Logger(ERROR, "File must be a YAML file");
            return false;
        }
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if(!fs.isOpened()){
            Logger(ERROR, "File could not be opened");
            return false;
        }

        this->gen.show = this->yamlReadBool(fs, "gen.show");
        this->gen.groundt = this->yamlReadBool(fs, "gen.groundt");
        this->gen.color_on = this->yamlReadBool(fs, "gen.color_on");
        this->gen.imu_on = this->yamlReadBool(fs, "gen.imu_on");
        this->gen.debug = this->yamlReadBool(fs, "gen.debug");

        this->gen.plot_tray = this->yamlReadBool(fs, "gen.plot_tray");
        this->gen.plot_vis_tray = this->yamlReadBool(fs, "gen.plot_vis_tray");
        this->gen.plot_imu_tray = this->yamlReadBool(fs, "gen.plot_imu_tray");
        this->gen.plot_gt_with_tray = this->yamlReadBool(fs, "gen.plot_gt_with_tray");
        this->gen.plot_gt_with_vis_tray = this->yamlReadBool(fs, "gen.plot_gt_with_vis_tray");
        this->gen.plot_gt_with_imu_tray = this->yamlReadBool(fs, "gen.plot_gt_with_imu_tray");
        this->gen.plot_2d = this->yamlReadBool(fs, "gen.plot_2d");
        this->gen.plot_imu = this->yamlReadBool(fs, "gen.plot_imu");
        this->gen.plot_rpy = this->yamlReadBool(fs, "gen.plot_rpy");
        this->gen.plot_vis_rpy = this->yamlReadBool(fs, "gen.plot_vis_rpy", false);
        this->gen.plot_gt_with_rpy = this->yamlReadBool(fs, "gen.plot_gt_with_rpy");
        this->gen.plot_gt_with_vis_rpy = this->yamlReadBool(fs, "gen.plot_gt_with_vis_rpy", false);

        fs["gen.input"] >> this->gen.input;
        fs["gen.output"] >> this->gen.output;

        if(this->toLower(this->gen.input).find(".bag") != std::string::npos){
            this->gen.type = SourceType::SOURCE_BAG;
        }else if(this->toLower(this->gen.input).find(".csv") != std::string::npos){
            this->configSetOnlyImu();
            this->gen.type = SourceType::SOURCE_CSV;
        }else if(this->toLower(this->gen.input).find(".mp4") != std::string::npos){
            this->configSetOnlyVis();
            this->gen.type = SourceType::SOURCE_MP4;
        }else if(this->toLower(this->gen.input).find("com") != std::string::npos){
            this->configSetOnlyImu();
            this->gen.type = SourceType::SOURCE_PORT;
        }else if(this->toLower(this->gen.input).find("rtsp://") != std::string::npos){
            this->gen.groundt = false;
            this->gen.type = SourceType::SOURCE_RTSP;
        }else{
            this->gen.type = SourceType::SOURCE_RSCAM;
        }

        fs["cam.fps"] >> this->cam.fps;
        fs["cam.width"] >> this->cam.w;
        fs["cam.height"] >> this->cam.h;
        fs["cam.K"] >> this->cam.K;
        fs["cam.D"] >> this->cam.D;

        fs["dep.min_dist"] >> this->dorb.min_dist;
        fs["dep.quality"] >> this->dorb.quality;
        fs["dep.lk_max_lvl"] >> this->dorb.lk_max_lvl;
        fs["dep.max_feat"] >> this->dorb.max_feat;
        fs["dep.min_pts"] >> this->dorb.min_pts;
        fs["dep.min_inliers"] >> this->dorb.min_inliers;
        fs["dep.lk_win_size"] >> this->dorb.lk_win_size;
        fs["dep.patch_r"] >> this->dorb.patch_r;

        this->dorb.min_dist = this->yamlReadDouble(fs, "orb.min_dist", this->dorb.min_dist); 
        this->dorb.quality = this->yamlReadDouble(fs, "orb.quality", this->dorb.quality); 
        this->dorb.lk_max_lvl = this->yamlReadDouble(fs, "orb.lk_max_lvl", this->dorb.lk_max_lvl);
        this->dorb.max_feat = this->yamlReadDouble(fs, "orb.max_feat", this->dorb.max_feat); 
        this->dorb.min_pts = this->yamlReadDouble(fs, "orb.min_pts", this->dorb.min_pts);
        this->dorb.min_inliers = this->yamlReadDouble(fs, "orb.min_inliers", this->dorb.min_inliers);
        this->dorb.lk_win_size = this->yamlReadDouble(fs, "orb.lk_win_size", this->dorb.lk_win_size);
        this->dorb.patch_r = this->yamlReadDouble(fs, "orb.patch_r", this->dorb.patch_r); 

        this->trk.min_ransac_points = this->yamlReadInt(fs, "trk.min_ransac_points", this->trk.min_ransac_points);
        this->trk.ransac_f_px = this->yamlReadDouble(fs, "trk.ransac_f_px", this->trk.ransac_f_px);
        this->trk.ransac_conf = this->yamlReadDouble(fs, "trk.ransac_conf", this->trk.ransac_conf);
        this->trk.fb_max_err_px = this->yamlReadDouble(fs, "trk.fb_max_err_px", this->trk.fb_max_err_px);
        this->trk.lk_iters = this->yamlReadInt(fs, "trk.lk_iters", this->trk.lk_iters);
        this->trk.lk_eps = this->yamlReadDouble(fs, "trk.lk_eps", this->trk.lk_eps);
        this->trk.gftt_block_size = this->yamlReadInt(fs, "trk.gftt_block_size", this->trk.gftt_block_size);
        this->trk.gftt_k = this->yamlReadDouble(fs, "trk.gftt_k", this->trk.gftt_k);

        this->vio.visual_step_scale = this->yamlReadDouble(fs, "vio.visual_step_scale", this->vio.visual_step_scale);
        this->vio.window_size = this->yamlReadInt(fs, "vio.window_size", this->vio.window_size);
        this->vio.min_parallax_deg = this->yamlReadDouble(fs, "vio.min_parallax_deg", this->vio.min_parallax_deg);
        this->vio.good_parallax_deg = this->yamlReadDouble(fs, "vio.good_parallax_deg", this->vio.good_parallax_deg);
        this->vio.min_track_length = this->yamlReadInt(fs, "vio.min_track_length", this->vio.min_track_length);
        this->vio.keyframe_max_age = this->yamlReadInt(fs, "vio.keyframe_max_age", this->vio.keyframe_max_age);
        this->vio.keyframe_parallax_deg = this->yamlReadDouble(fs, "vio.keyframe_parallax_deg", this->vio.keyframe_parallax_deg);
        this->vio.triang_min_points = this->yamlReadInt(fs, "vio.triang_min_points", this->vio.triang_min_points);
        this->vio.triang_max_reproj_px = this->yamlReadDouble(fs, "vio.triang_max_reproj_px", this->vio.triang_max_reproj_px);
        this->vio.pnp_min_points = this->yamlReadInt(fs, "vio.pnp_min_points", this->vio.pnp_min_points);
        this->vio.pnp_reproj_px = this->yamlReadDouble(fs, "vio.pnp_reproj_px", this->vio.pnp_reproj_px);
        this->vio.pnp_conf = this->yamlReadDouble(fs, "vio.pnp_conf", this->vio.pnp_conf);
        this->vio.landmark_min_obs = this->yamlReadInt(fs, "vio.landmark_min_obs", this->vio.landmark_min_obs);
        this->vio.landmark_max_age = this->yamlReadInt(fs, "vio.landmark_max_age", this->vio.landmark_max_age);
        this->vio.landmark_max_reproj_px = this->yamlReadDouble(fs, "vio.landmark_max_reproj_px", this->vio.landmark_max_reproj_px);
        this->vio.pnp_use_pose = this->yamlReadBool(fs, "vio.pnp_use_pose", this->vio.pnp_use_pose);
        this->vio.pose_refine_enable = this->yamlReadBool(fs, "vio.pose_refine_enable", this->vio.pose_refine_enable);
        this->vio.pose_refine_min_points = this->yamlReadInt(fs, "vio.pose_refine_min_points", this->vio.pose_refine_min_points);
        this->vio.pose_refine_max_iters = this->yamlReadInt(fs, "vio.pose_refine_max_iters", this->vio.pose_refine_max_iters);
        this->vio.pose_refine_reproj_px = this->yamlReadDouble(fs, "vio.pose_refine_reproj_px", this->vio.pose_refine_reproj_px);
        this->vio.pose_refine_huber_px = this->yamlReadDouble(fs, "vio.pose_refine_huber_px", this->vio.pose_refine_huber_px);
        this->vio.pose_refine_stop_dx = this->yamlReadDouble(fs, "vio.pose_refine_stop_dx", this->vio.pose_refine_stop_dx);
        this->vio.local_ba_enable = this->yamlReadBool(fs, "vio.local_ba_enable", this->vio.local_ba_enable);
        this->vio.local_ba_min_landmarks = this->yamlReadInt(fs, "vio.local_ba_min_landmarks", this->vio.local_ba_min_landmarks);
        this->vio.local_ba_max_landmarks = this->yamlReadInt(fs, "vio.local_ba_max_landmarks", this->vio.local_ba_max_landmarks);
        this->vio.local_ba_max_iters = this->yamlReadInt(fs, "vio.local_ba_max_iters", this->vio.local_ba_max_iters);
        this->vio.local_ba_reproj_px = this->yamlReadDouble(fs, "vio.local_ba_reproj_px", this->vio.local_ba_reproj_px);
        this->vio.local_ba_huber_px = this->yamlReadDouble(fs, "vio.local_ba_huber_px", this->vio.local_ba_huber_px);
        this->vio.local_ba_stop_dx = this->yamlReadDouble(fs, "vio.local_ba_stop_dx", this->vio.local_ba_stop_dx);
        this->vio.local_ba_min_improve_px = this->yamlReadDouble(fs, "vio.local_ba_min_improve_px", this->vio.local_ba_min_improve_px);
        this->vio.local_ba_accept_rot_deg = this->yamlReadDouble(fs, "vio.local_ba_accept_rot_deg", this->vio.local_ba_accept_rot_deg);
        this->vio.local_ba_accept_pos_m = this->yamlReadDouble(fs, "vio.local_ba_accept_pos_m", this->vio.local_ba_accept_pos_m);
        this->vio.clone_factor_enable = this->yamlReadBool(fs, "vio.clone_factor_enable", this->vio.clone_factor_enable);
        this->vio.fuse_enable = this->yamlReadBool(fs, "vio.fuse_enable", this->vio.fuse_enable);
        this->vio.fuse_min_inliers = this->yamlReadInt(fs, "vio.fuse_min_inliers", this->vio.fuse_min_inliers);
        this->vio.fuse_max_reproj_px = this->yamlReadDouble(fs, "vio.fuse_max_reproj_px", this->vio.fuse_max_reproj_px);
        this->vio.fuse_min_parallax_deg = this->yamlReadDouble(fs, "vio.fuse_min_parallax_deg", this->vio.fuse_min_parallax_deg);
        this->vio.fuse_pos_gain = this->yamlReadDouble(fs, "vio.fuse_pos_gain", this->vio.fuse_pos_gain);
        this->vio.fuse_vel_gain = this->yamlReadDouble(fs, "vio.fuse_vel_gain", this->vio.fuse_vel_gain);
        this->vio.fuse_ori_gain = this->yamlReadDouble(fs, "vio.fuse_ori_gain", this->vio.fuse_ori_gain);
        this->vio.fuse_max_pos_corr_m = this->yamlReadDouble(fs, "vio.fuse_max_pos_corr_m", this->vio.fuse_max_pos_corr_m);
        this->vio.fuse_max_vel_corr_ms = this->yamlReadDouble(fs, "vio.fuse_max_vel_corr_ms", this->vio.fuse_max_vel_corr_ms);
        this->vio.fuse_max_ori_corr_deg = this->yamlReadDouble(fs, "vio.fuse_max_ori_corr_deg", this->vio.fuse_max_ori_corr_deg);

        fs["imu.accfps"] >> this->imu.acc_fps;
        fs["imu.gyrfps"] >> this->imu.gyr_fps;
        this->imu.batch_ms = this->yamlReadDouble(fs, "imu.batch_ms", this->imu.batch_ms);
        this->imu.stationary_enable = this->yamlReadBool(fs, "imu.stationary_enable", this->imu.stationary_enable);
        this->imu.zupt_enable = this->yamlReadBool(fs, "imu.zupt_enable", this->imu.zupt_enable);
        this->imu.bias_refine_enable = this->yamlReadBool(fs, "imu.bias_refine_enable", this->imu.bias_refine_enable);
        this->imu.stationary_min_samples = this->yamlReadInt(fs, "imu.stationary_min_samples", this->imu.stationary_min_samples);
        this->imu.stationary_gyro_mean_max = this->yamlReadDouble(fs, "imu.stationary_gyro_mean_max", this->imu.stationary_gyro_mean_max);
        this->imu.stationary_acc_mean_tol = this->yamlReadDouble(fs, "imu.stationary_acc_mean_tol", this->imu.stationary_acc_mean_tol);
        this->imu.stationary_gyro_std_max = this->yamlReadDouble(fs, "imu.stationary_gyro_std_max", this->imu.stationary_gyro_std_max);
        this->imu.stationary_acc_std_max = this->yamlReadDouble(fs, "imu.stationary_acc_std_max", this->imu.stationary_acc_std_max);
        this->imu.stationary_bias_alpha = this->yamlReadDouble(fs, "imu.stationary_bias_alpha", this->imu.stationary_bias_alpha);
        this->imu.gv = this->yamlReadVec3(fs, "imu.gv", this->imu.gv);
        this->imu.bg = this->yamlReadVec3(fs, "imu.bg", this->imu.bg);
        this->imu.ba = this->yamlReadVec3(fs, "imu.ba", this->imu.ba);
        this->imu.allangx = this->yamlReadVec3(fs, "imu.allangx", this->imu.allangx);
        this->imu.allangy = this->yamlReadVec3(fs, "imu.allangy", this->imu.allangy);
        this->imu.allangz = this->yamlReadVec3(fs, "imu.allangz", this->imu.allangz);
        this->imu.allanax = this->yamlReadVec3(fs, "imu.allanax", this->imu.allanax);
        this->imu.allanay = this->yamlReadVec3(fs, "imu.allanay", this->imu.allanay);
        this->imu.allanaz = this->yamlReadVec3(fs, "imu.allanaz", this->imu.allanaz);
        this->imu.small_angle = this->yamlReadDouble(fs, "imu.small_angle", this->imu.small_angle);
        if (!fs["imu.tocolor"].empty()) {
            fs["imu.tocolor"] >> this->imu.T;
            this->imu.T.convertTo(this->imu.T, CV_64F);
        }

        return true;
    }

private:
    std::string toLower(std::string s){
        std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    bool yamlReadBool(const cv::FileStorage& fs, const char * key){ 
        return yamlReadBool(fs, key, false);
    }

    bool yamlReadBool(const cv::FileStorage& fs, const char * key, bool fallback){
        cv::FileNode node = fs[key];
        if (node.empty()) {
            return fallback;
        }

        if (node.isString()) {
            std::string temp;
            node >> temp;
            temp = toLower(temp);
            if (temp == "true" || temp == "1" || temp == "yes" || temp == "on") return true;
            if (temp == "false" || temp == "0" || temp == "no" || temp == "off") return false;
            return fallback;
        }

        const double numeric = static_cast<double>(node.real());
        if (std::isfinite(numeric)) {
            return std::abs(numeric) > 0.5;
        }
        return fallback;
    }

    double yamlReadDouble(const cv::FileStorage& fs, const char * key, double fallback = 0.0){
        cv::FileNode node = fs[key];
        if (node.empty()) {
            return fallback;
        }

        double value = fallback;
        node >> value;
        return std::isfinite(value) ? value : fallback;
    }

    int yamlReadInt(const cv::FileStorage& fs, const char * key, int fallback = 0){
        cv::FileNode node = fs[key];
        if (node.empty()) {
            return fallback;
        }

        int value = fallback;
        node >> value;
        return value;
    }

    vec3 yamlReadVec3(const cv::FileStorage& fs, const char * key, const vec3& fallback = vec3::Zero()){
        cv::FileNode node = fs[key];
        if (node.empty()) {
            return fallback;
        }

        cv::Mat temp;
        node >> temp;
        if (temp.empty()) {
            return fallback;
        }

        cv::Mat temp64;
        temp.convertTo(temp64, CV_64F);
        temp64 = temp64.reshape(1, 1);
        if (temp64.total() < 3) {
            return fallback;
        }

        return vec3(temp64.at<double>(0,0), temp64.at<double>(0,1), temp64.at<double>(0,2));
    }

    static const char* boolText(bool v){
        return v ? "ACTIVO" : "NO ACTIVO";
    }

    static const char* sourceTypeToString(SourceType t){
        switch(t){
            case SOURCE_RSCAM: return "SOURCE_RSCAM";
            case SOURCE_BAG:   return "SOURCE_BAG";
            case SOURCE_MP4:   return "SOURCE_MP4";
            case SOURCE_RTSP:  return "SOURCE_RTSP";
            case SOURCE_CSV:   return "SOURCE_CSV";
            case SOURCE_PORT:  return "SOURCE_PORT";
            default:           return "SOURCE_UNKNOWN";
        }
    }

    static std::string matToString(const cv::Mat& m, int precision = 6){
        if (m.empty()) return "    [EMPTY]";

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(precision);

        for (int r = 0; r < m.rows; ++r) {
            oss << "    [ ";
            for (int c = 0; c < m.cols; ++c) {
                if (m.type() == CV_64F) {
                    oss << std::setw(12) << m.at<double>(r, c);
                } else if (m.type() == CV_32F) {
                    oss << std::setw(12) << m.at<float>(r, c);
                } else if (m.type() == CV_32S) {
                    oss << std::setw(12) << m.at<int>(r, c);
                } else if (m.type() == CV_8U) {
                    oss << std::setw(12) << static_cast<int>(m.at<unsigned char>(r, c));
                } else {
                    oss << std::setw(12) << "?";
                }

                if (c < m.cols - 1) oss << ", ";
            }
            oss << " ]";
            if (r < m.rows - 1) oss << "\n";
        }
        return oss.str();
    }
};

struct ImuSample {
    double ts = 0.0;
    double dt = 0.0;
    vec3 vgyr;
    vec3 vacc;
};

struct SourceIn {
    cv::Mat frame;
    double frame_tsms = 0.0;
    double frame_dtms = 0.0;

    cv::Mat depth;
    double depth_tsms = 0.0;

    std::deque<ImuSample> imu;

    static std::string VecToStr(const vec3& v) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(3)
           << "[" << v.x() << "," << v.y() << "," << v.z() << "]";
        return os.str();
    }

    void print() const {
        std::cout << std::fixed << std::setprecision(3);

        std::cout << "frame: " << frame.cols << "x" << frame.rows << " ts=" << frame_tsms << " dt=" << frame_dtms << "\n";

        std::cout << "depth: " << depth.cols << "x" << depth.rows << " ts=" << depth_tsms << "\n";

        vec3 gyr_mean = vec3::Zero();
        vec3 acc_mean = vec3::Zero();
        double dt_sum = 0.0;

        for (const ImuSample& s : imu) {
            gyr_mean += s.vgyr;
            acc_mean += s.vacc;
            dt_sum += s.dt;
        }

        if (!imu.empty()) {
            const double n = static_cast<double>(imu.size());
            gyr_mean /= n;
            acc_mean /= n;
        }

        std::cout << "imu(" << imu.size() << ")" << " t0=" << (imu.empty() ? 0.0 : imu.front().ts) << " t1=" << (imu.empty() ? 0.0 : imu.back().ts) << " dt_sum=" << dt_sum << " gyr_mean=" << VecToStr(gyr_mean) << " acc_mean=" << VecToStr(acc_mean) << "\n";
    }
};

struct DebugState {
    // Visual debug pose/state
    vec3 vis_xyz = vec3::Zero();
    vec3 vis_rpy = vec3::Zero();
    float scale = 1.0f;
    bool vio_valid = false;
    bool vio_rel_valid = false;
    bool vio_trans_valid = false;
    bool vio_pnp_valid = false;
    bool vio_refined = false;
    bool vio_local_ba = false;
    bool vio_clone_factor = false;
    bool vio_fused = false;
    int vio_matches = 0;
    int vio_tracked = 0;
    int vio_inliers = 0;
    int vio_pnp_corr = 0;
    int vio_kf_age = 0;
    int vio_window_kfs = 0;
    int vio_ref_age = 0;
    double vio_kf_parallax_deg = 0.0;
    int vio_init_tracks = 0;
    int vio_pose_only_tracks = 0;
    int vio_explo_tracks = 0;
    int vio_triangulated = 0;
    int vio_pnp_inliers = 0;
    int vio_refine_inliers = 0;
    int vio_ba_landmarks = 0;
    int vio_ba_obs = 0;
    int vio_ba_inliers = 0;
    int vio_clone_features = 0;
    int vio_clone_obs = 0;
    int vio_clone_inliers = 0;
    int vio_landmarks = 0;
    double vio_reproj_px = 0.0;
    double vio_refine_reproj_px = 0.0;
    double vio_ba_reproj_px = 0.0;
    double vio_clone_reproj_px = 0.0;
    double vio_fuse_pos_res_m = 0.0;
    double vio_fuse_vel_res_ms = 0.0;
    double vio_fuse_ori_res_deg = 0.0;
    
    // IMU-only global pose for plots/logger.
    // These fields are what "plot_imu_tray" and "plot_rpy" should use.
    vec3 imu_rpy = vec3::Zero();
    vec3 imu_xyz = vec3::Zero();

    // Local preintegrated IMU delta used by the visual updater.
    vec3 imu_dp = vec3::Zero();
    vec3 imu_dv = vec3::Zero();
    vec3 imu_drpy = vec3::Zero();
    quat imu_dq = quat::Identity();
    bool imu_stationary = false;

    // RAW IMU
    vec3 acc_ms2 = vec3::Zero();
    vec3 gyr_rads = vec3::Zero();
};

struct StateOut{
    double ts_ms = 0.0;       // Time
    double dt = 0.0;

    vec3 pos_m = vec3::Zero();         // final state (Pos)
    vec3 vel_ms = vec3::Zero();
    vec3 acc_cal_ms2 = vec3::Zero();   // IMU after calibration

    vec3 rpy_rad = vec3::Zero();       // final state (Ori)
    quat quat_rad = quat::Identity();
    vec3 gyr_cal_rads = vec3::Zero();

    vec3 gravv = vec3::Zero();

    vec3 posgt_m = vec3::Zero();       // Ground truth
    vec3 origt_rad = vec3::Zero();       // Ground truth

    Eigen::Matrix<double,15,27> H;

    DebugState deb;     // Debug metrics

    // const int IDX_Q = 0;
    // const int IDX_P = 4;
    // const int IDX_V = 7;
    // const int IDX_BG = 10;
    // const int IDX_BA = 13;
    // vec16 pint_imus;    // IMU state [q,p,v,bg,ba] < R16

    

    std::vector<double> toVector(bool include_debug = false){
        std::vector<double> out;

        out.push_back(this->ts_ms);
        out.push_back(this->dt);

        out.push_back(this->pos_m.x());
        out.push_back(this->pos_m.y());
        out.push_back(this->pos_m.z());

        out.push_back(this->vel_ms.x());
        out.push_back(this->vel_ms.y());
        out.push_back(this->vel_ms.z());

        out.push_back(this->acc_cal_ms2.x());
        out.push_back(this->acc_cal_ms2.y());
        out.push_back(this->acc_cal_ms2.z());

        out.push_back(this->rpy_rad.x());
        out.push_back(this->rpy_rad.y());
        out.push_back(this->rpy_rad.z());

        out.push_back(this->quat_rad.w());
        out.push_back(this->quat_rad.x());
        out.push_back(this->quat_rad.y());
        out.push_back(this->quat_rad.z());

        out.push_back(this->gyr_cal_rads.x());
        out.push_back(this->gyr_cal_rads.y());
        out.push_back(this->gyr_cal_rads.z());

        out.push_back(this->posgt_m.x());
        out.push_back(this->posgt_m.y());
        out.push_back(this->posgt_m.z());

        out.push_back(this->origt_rad.x());
        out.push_back(this->origt_rad.y());
        out.push_back(this->origt_rad.z());

        if (include_debug) {
            out.push_back(this->deb.vis_xyz.x());
            out.push_back(this->deb.vis_xyz.y());
            out.push_back(this->deb.vis_xyz.z());

            out.push_back(this->deb.vis_rpy.x());
            out.push_back(this->deb.vis_rpy.y());
            out.push_back(this->deb.vis_rpy.z());

            out.push_back(static_cast<double>(this->deb.scale));
            out.push_back(static_cast<double>(this->deb.vio_valid));
            out.push_back(static_cast<double>(this->deb.vio_rel_valid));
            out.push_back(static_cast<double>(this->deb.vio_trans_valid));
            out.push_back(static_cast<double>(this->deb.vio_pnp_valid));
            out.push_back(static_cast<double>(this->deb.vio_refined));
            out.push_back(static_cast<double>(this->deb.vio_local_ba));
            out.push_back(static_cast<double>(this->deb.vio_clone_factor));
            out.push_back(static_cast<double>(this->deb.vio_fused));
            out.push_back(static_cast<double>(this->deb.vio_matches));
            out.push_back(static_cast<double>(this->deb.vio_tracked));
            out.push_back(static_cast<double>(this->deb.vio_inliers));
            out.push_back(static_cast<double>(this->deb.vio_pnp_corr));
            out.push_back(static_cast<double>(this->deb.vio_kf_age));
            out.push_back(static_cast<double>(this->deb.vio_window_kfs));
            out.push_back(static_cast<double>(this->deb.vio_ref_age));
            out.push_back(this->deb.vio_kf_parallax_deg);
            out.push_back(static_cast<double>(this->deb.vio_init_tracks));
            out.push_back(static_cast<double>(this->deb.vio_pose_only_tracks));
            out.push_back(static_cast<double>(this->deb.vio_explo_tracks));
            out.push_back(static_cast<double>(this->deb.vio_triangulated));
            out.push_back(static_cast<double>(this->deb.vio_pnp_inliers));
            out.push_back(static_cast<double>(this->deb.vio_refine_inliers));
            out.push_back(static_cast<double>(this->deb.vio_ba_landmarks));
            out.push_back(static_cast<double>(this->deb.vio_ba_obs));
            out.push_back(static_cast<double>(this->deb.vio_ba_inliers));
            out.push_back(static_cast<double>(this->deb.vio_clone_features));
            out.push_back(static_cast<double>(this->deb.vio_clone_obs));
            out.push_back(static_cast<double>(this->deb.vio_clone_inliers));
            out.push_back(static_cast<double>(this->deb.vio_landmarks));
            out.push_back(this->deb.vio_reproj_px);
            out.push_back(this->deb.vio_refine_reproj_px);
            out.push_back(this->deb.vio_ba_reproj_px);
            out.push_back(this->deb.vio_clone_reproj_px);
            out.push_back(this->deb.vio_fuse_pos_res_m);
            out.push_back(this->deb.vio_fuse_vel_res_ms);
            out.push_back(this->deb.vio_fuse_ori_res_deg);

            out.push_back(this->deb.imu_rpy.x());
            out.push_back(this->deb.imu_rpy.y());
            out.push_back(this->deb.imu_rpy.z());

            out.push_back(this->deb.imu_xyz.x());
            out.push_back(this->deb.imu_xyz.y());
            out.push_back(this->deb.imu_xyz.z());

            out.push_back(static_cast<double>(this->deb.imu_stationary));

            out.push_back(this->deb.imu_dp.x());
            out.push_back(this->deb.imu_dp.y());
            out.push_back(this->deb.imu_dp.z());

            out.push_back(this->deb.imu_dv.x());
            out.push_back(this->deb.imu_dv.y());
            out.push_back(this->deb.imu_dv.z());

            out.push_back(this->deb.imu_drpy.x());
            out.push_back(this->deb.imu_drpy.y());
            out.push_back(this->deb.imu_drpy.z());

            out.push_back(this->deb.imu_dq.w());
            out.push_back(this->deb.imu_dq.x());
            out.push_back(this->deb.imu_dq.y());
            out.push_back(this->deb.imu_dq.z());

            out.push_back(this->deb.acc_ms2.x());
            out.push_back(this->deb.acc_ms2.y());
            out.push_back(this->deb.acc_ms2.z());

            out.push_back(this->deb.gyr_rads.x());
            out.push_back(this->deb.gyr_rads.y());
            out.push_back(this->deb.gyr_rads.z());
        }

        return out;
    }
};

