#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
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
typedef Eigen::Vector2d vec2;
typedef Eigen::Vector4d vec4;
typedef Eigen::Quaterniond quat;
typedef Eigen::Matrix<double,16,1> vec16;
typedef Eigen::Matrix<double,10,1> vec10;
typedef Eigen::Matrix3d mat3;
typedef Eigen::Matrix<double, 6, 6> mat6;
typedef Eigen::Matrix<double, 9, 9> mat9;
typedef Eigen::Matrix<double, 9, 6> mat96;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> SqrMatrixType;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
const double null = std::numeric_limits<double>::quiet_NaN();


#define UDP_SEND_PORT 5001
#define UDP_RECV_PORT 5002
#define UDP_IP L"192.168.1.145"
#define RTSP_PORT 8554
#define RTSP_IP "192.168.1.145"
#define RTSP_MOUNT "realsense"
#define MODEL_WIDTH 504
#define MODEL_HEIGHT 280

enum SourceType{
    SOURCE_RSCAM = 0,
    SOURCE_BAG,
    SOURCE_MP4,
    SOURCE_RTSP,
    SOURCE_CSV,
    SOURCE_PORT,

    SOURCE_COUNT
};

enum Da3Model {
    DA3_MODEL_SMALL = 0,
    DA3_MODEL_BASE,

    DA3_MODEL_COUNT
};

struct imuCal{

    double fps = 200.0;

    vec3 ba = vec3::Zero();                 // Static gyr bias
    vec3 bg = vec3::Zero();                 // Static acc bias

    vec3 allanaN = vec3::Zero();            // aN(x,y,z)
    vec3 allangN = vec3::Zero();            // gN(x,y,z)
    vec3 allanaK = vec3::Zero();            // aK(x,y,z)
    vec3 allangK = vec3::Zero();            // gK(x,y,z)


    double g = 9.808;                         // gravity magnitude (positive magnitude as expected by RVIO2)
    cv::Mat T_ci;                           // Imu to camera transformation

};
struct General{
    SourceType type;
    std::string input;
    std::string output;

    bool depth_on;              // Turn on/off ground truth depth based
    bool color_on;              // Turn on/off color stream
    bool imu_on;                // Turn on/off inertial stream
    bool show;                  // Show image
    bool debug;                 // Turn on/off debug info

    bool plot_imu;              // Plot Accel and Gyro calibrated and raw 

    bool plot_tray;             // Plot Trayectory
    bool plot_vis_tray;         // Plot Visual only trayectory
    bool plot_imu_tray;         // Plot Inertial only trayectory
    bool plot_height;           // Plot height with its trayectory

    bool plot_rpy;              // Plot IMU output as RPY
    bool plot_vis_rpy;          // Plot visual-only RPY plot
    bool plot_imu_rpy;          // Plot inertial-only RPY plot

    bool plot_dpos;             // Plot pre-int pos solution
    bool plot_dvel;             // plot pre-int vel solution
    bool plot_da3;              // Plot DA3 avoidance angle and magnitude
};
struct Camera {
    int width = 0;
    int height = 0;
    double fps = 0.0;
    bool is_rgb = false;
    // RVIO2 measurement noise defaults in normalized image coordinates.
    double spx = 0.002180293;
    double spy = 0.002186767;

    cv::Mat K;
    cv::Mat D;

    double getFx() { return this->K.at<double>(0, 0);}
    double getFy() { return this->K.at<double>(1, 1);}
    double getCx() { return this->K.at<double>(0, 2);}
    double getCy() { return this->K.at<double>(1, 2);}

};

struct Vio {
    double feat_mindist = 8.0;
    double feat_quat = 1e-2;
    double feat_bsizex = 150.0;
    double feat_bsizey = 120.0;
    int feat_max = 1200;
    int rsac_it = 16;
    bool sampson_on = true;
    double sampson_threserr = 1e-6;
    double sampson_algerr = 1e-3;
    double good_para = 5.0;
    bool track_filter_on = false;
    bool track_en_equalizer = true;
    int track_maxlength = 18;
    int track_minlength = 6;
    bool track_show = false;
    bool track_shownew = false;
    int slam_pts = 0;
    double ang_ths = 0.3;
    double dis_ths = 0.01;
    bool en_align = false;
};
struct Da3Config {
    bool enabled = false;
    bool show_window = false;
    Da3Model model = DA3_MODEL_SMALL;
    std::string model_path = "../../model/DA3METRIC-SMALL.onnx";
    int input_width = 504;
    int input_height = 280;
    double min_valid_depth = 0.05;
    double max_valid_depth = 20.0;
    double evade_distance = 1.5;
    double direction_deadband = 1e-5;
    double smooth_alpha = 0.7;
    double debug_arrow_gain = 10000.0;
    int sector_count = 9;
    int morph_kernel_size = 5;
    double frontal_roi_width_ratio = 0.40;
    double frontal_roi_height_ratio = 0.45;
    double guidance_roi_width_ratio = 0.90;
    double guidance_roi_height_ratio = 0.55;
    double close_mask_threshold = 0.60;
    double activate_score_threshold = 0.28;
    double clear_score_threshold = 0.18;
    double min_close_area_ratio = 0.03;
    double min_blob_area_ratio = 0.015;
    double emergency_close_area_ratio = 0.08;
    double emergency_blob_area_ratio = 0.05;
    double score_weight_mean = 0.35;
    double score_weight_area = 0.25;
    double score_weight_blob = 0.25;
    double score_weight_p20 = 0.15;
    double score_weight_peak = 0.20;
    double frontal_peak_percentile = 0.05;
    double peak_score_threshold = 0.92;
    double sector_close_penalty = 0.75;
    double free_space_power = 2.0;
};
struct Config{
    General gen;
    Camera cam;
    imuCal imu;
    Vio vio;
    Da3Config da3;

public:

    void print(std::ostream& os = std::cout) const {
        os << "\n========================================\n";
        os << "              CONFIG PRINT              \n";
        os << "========================================\n";

        os << "\n[GENERAL]\n";
        os << "  Source type   : " << sourceTypeToString(this->gen.type) << "\n";
        os << "  Input         : " << this->gen.input << "\n";
        os << "  Output        : " << this->gen.output << "\n";
        os << "  Ground truth  : " << std::string(this->gen.depth_on ? "ON" : "OFF") << "\n";
        os << "  Color         : " << std::string(this->gen.color_on ? "ON" : "OFF") << "\n";
        os << "  IMU           : " << std::string(this->gen.imu_on ? "ON" : "OFF") << "\n";
        os << "  Show          : " << std::string(this->gen.show ? "ON" : "OFF") << "\n";
        os << "  Debug         : " << std::string(this->gen.debug ? "ON" : "OFF") << "\n";

        os << "\n[CAMERA]\n";
        os << "  Width         : " << this->cam.width << "\n";
        os << "  Height        : " << this->cam.height << "\n";
        os << "  FPS           : " << this->cam.fps << "\n";
        os << "  K             :\n" << matToString(this->cam.K, 10) << "\n";
        os << "  D             :\n" << matToString(this->cam.D, 10) << "\n";

        os << "\n[VIO]\n";
        os << "FEATURE MIN. DISTANCE     :" << this->vio.feat_mindist << "\n"; 
        os << "FEATURE QUALITY           :" << this->vio.feat_quat << "\n";
        os << "FEATURE BLOCK SIZE X      :" << this->vio.feat_bsizex << "\n";
        os << "FEATURE BLOCK SIZE Y      :" << this->vio.feat_bsizey << "\n";
        os << "FEATURE MAX ELEMENTS      :" << this->vio.feat_max << "\n";
        os << "RANSANC ITERATIONS        :" << this->vio.rsac_it << "\n";
        os << "SAMPSON THRESHOLD ERR     :" << this->vio.sampson_threserr << "\n";
        os << "SAMPSON THRESHOLD ALG ERR :" << this->vio.sampson_algerr << "\n";
        os << "GOOD PARALLAX             :" << this->vio.good_para << "\n";
        os << "TRACKER MAX LENGTH        :" << this->vio.track_maxlength << "\n";
        os << "TRACKER MIN LENGTH        :" << this->vio.track_minlength << "\n";
        os << "SLAM POINTS               :" << this->vio.slam_pts << "\n";
        os << "ANGLE THRESHOLD           :" << this->vio.ang_ths << "\n";
        os << "DISTANCE THRESHOLD        :" << this->vio.dis_ths << "\n";
        os << "TRACKER EQUALIZER ON      :" << std::string(this->vio.track_en_equalizer ? "ON" : "OFF") << "\n";
        os << "TRACKER FILTER ON         :" << std::string(this->vio.track_filter_on ? "ON" : "OFF") << "\n";
        os << "TRACKER SHOW              :" << std::string(this->vio.track_show ? "ON" : "OFF") << "\n";
        os << "TRACKER SHOW NEW          :" << std::string(this->vio.track_shownew ? "ON" : "OFF") << "\n";
        os << "SAMPSON ON                :" << std::string(this->vio.sampson_on ? "ON" : "OFF") << "\n";
        os << "ALIGN ENABLE              :" << std::string(this->vio.en_align ? "ON" : "OFF") << "\n";

        os << "\n[DA3]\n";
        os << "  Enabled       : " << std::string(this->da3.enabled ? "ON" : "OFF") << "\n";
        os << "  Show window   : " << std::string(this->da3.show_window ? "ON" : "OFF") << "\n";
        os << "  Model         : " << da3ModelToString(this->da3.model) << "\n";
        os << "  Model path    : " << this->da3.model_path << "\n";
        os << "  Input size    : " << this->da3.input_width << "x" << this->da3.input_height << "\n";
        os << "  Min depth     : " << this->da3.min_valid_depth << "\n";
        os << "  Max depth     : " << this->da3.max_valid_depth << "\n";
        os << "  Evade dist.   : " << this->da3.evade_distance << "\n";
        os << "  Deadband      : " << this->da3.direction_deadband << "\n";
        os << "  Smooth alpha  : " << this->da3.smooth_alpha << "\n";
        os << "  Arrow gain    : " << this->da3.debug_arrow_gain << "\n";
        os << "  Sector count  : " << this->da3.sector_count << "\n";
        os << "  Morph kernel  : " << this->da3.morph_kernel_size << "\n";
        os << "  Frontal ROI   : " << this->da3.frontal_roi_width_ratio << "x" << this->da3.frontal_roi_height_ratio << "\n";
        os << "  Guidance ROI  : " << this->da3.guidance_roi_width_ratio << "x" << this->da3.guidance_roi_height_ratio << "\n";
        os << "  Close thr.    : " << this->da3.close_mask_threshold << "\n";
        os << "  Score in/out  : " << this->da3.activate_score_threshold << " / " << this->da3.clear_score_threshold << "\n";
        os << "  Area thr.     : " << this->da3.min_close_area_ratio << " / " << this->da3.emergency_close_area_ratio << "\n";
        os << "  Blob thr.     : " << this->da3.min_blob_area_ratio << " / " << this->da3.emergency_blob_area_ratio << "\n";
        os << "  Peak cfg.     : " << this->da3.frontal_peak_percentile << " / " << this->da3.peak_score_threshold << "\n";
        os << "  Score wts.    : " << this->da3.score_weight_mean << ", " << this->da3.score_weight_area << ", " << this->da3.score_weight_blob << ", " << this->da3.score_weight_p20 << ", " << this->da3.score_weight_peak << "\n";
        os << "  Sector pen.   : " << this->da3.sector_close_penalty << "\n";
        os << "  Free pow.     : " << this->da3.free_space_power << "\n";

        os << "\n[PLOTTERS]\n";
        os << "IMU         :" << std::string(this->gen.plot_imu      ? "ON" : "OFF") << "\n"; 
        os << "TRAY        :" << std::string(this->gen.plot_tray     ? "ON" : "OFF") << "\n"; 
        os << "VIS_TRAY    :" << std::string(this->gen.plot_vis_tray ? "ON" : "OFF") << "\n"; 
        os << "IMU_TRAY    :" << std::string(this->gen.plot_imu_tray ? "ON" : "OFF") << "\n"; 
        os << "HEIGHT      :" << std::string(this->gen.plot_height   ? "ON" : "OFF") << "\n"; 
        os << "RPY         :" << std::string(this->gen.plot_rpy      ? "ON" : "OFF") << "\n"; 
        os << "VIS_RPY     :" << std::string(this->gen.plot_vis_rpy  ? "ON" : "OFF") << "\n"; 
        os << "IMU_RPY     :" << std::string(this->gen.plot_imu_rpy  ? "ON" : "OFF") << "\n"; 
        os << "DPOS        :" << std::string(this->gen.plot_dpos     ? "ON" : "OFF") << "\n"; 
        os << "DVEL        :" << std::string(this->gen.plot_dvel     ? "ON" : "OFF") << "\n"; 
        os << "DA3         :" << std::string(this->gen.plot_da3      ? "ON" : "OFF") << "\n"; 

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
        fs["gen.input"] >> this->gen.input;
        fs["gen.output"] >> this->gen.output;

        this->gen.show                  = this->yamlReadBool(fs, "gen.show", false);
        this->gen.depth_on              = this->yamlReadBool(fs, "gen.depth_on", false);
        this->gen.color_on              = this->yamlReadBool(fs, "gen.color_on", true);
        this->gen.imu_on                = this->yamlReadBool(fs, "gen.imu_on", true);
        this->gen.debug                 = this->yamlReadBool(fs, "gen.debug", false);

        this->gen.plot_imu          = this->yamlReadBool(fs, "gen.plot_imu", false);
        this->gen.plot_tray         = this->yamlReadBool(fs, "gen.plot_tray", false);
        this->gen.plot_vis_tray     = this->yamlReadBool(fs, "gen.plot_vis_tray", false);
        this->gen.plot_imu_tray     = this->yamlReadBool(fs, "gen.plot_imu_tray", false);
        this->gen.plot_height       = this->yamlReadBool(fs, "gen.plot_height", false);
        this->gen.plot_rpy          = this->yamlReadBool(fs, "gen.plot_rpy", false);
        this->gen.plot_vis_rpy      = this->yamlReadBool(fs, "gen.plot_vis_rpy", false);
        this->gen.plot_imu_rpy      = this->yamlReadBool(fs, "gen.plot_imu_rpy", false);
        this->gen.plot_dpos         = this->yamlReadBool(fs, "gen.plot_dpos", false);
        this->gen.plot_dvel         = this->yamlReadBool(fs, "gen.plot_dvel", false);
        this->gen.plot_da3          = this->yamlReadBool(fs, "gen.plot_da3", false);

        if(this->toLower(this->gen.input).find(".bag") != std::string::npos){
            this->gen.type = SourceType::SOURCE_BAG;
        }else if(this->toLower(this->gen.input).find(".csv") != std::string::npos){
            this->gen.imu_on = true; this->gen.color_on = false; this->gen.depth_on = false;
            this->gen.type = SourceType::SOURCE_CSV;
        }else if(this->toLower(this->gen.input).find("com") != std::string::npos){
            this->gen.imu_on = true; this->gen.color_on = false; this->gen.depth_on = false;
            this->gen.type = SourceType::SOURCE_PORT;
        }else if(this->toLower(this->gen.input).find("rtsp://") != std::string::npos){
            this->gen.depth_on = false;
            this->gen.type = SourceType::SOURCE_RTSP;
        }else{
            this->gen.type = SourceType::SOURCE_RSCAM;
        }

        this->cam.fps    = this->yamlReadDouble(fs, "cam.fps", this->cam.fps);
        this->cam.width  = this->yamlReadInt(fs, "cam.width", this->cam.width);
        this->cam.height = this->yamlReadInt(fs, "cam.height", this->cam.height);
        this->cam.spx    = this->yamlReadDouble(fs, "cam.spx", this->cam.spx);
        this->cam.spy    = this->yamlReadDouble(fs, "cam.spy", this->cam.spy);
        fs["cam.K"]      >> this->cam.K;
        fs["cam.D"]      >> this->cam.D;
        

        this->imu.fps = this->yamlReadDouble(fs, "imu.fps", this->imu.fps);
        this->imu.g = std::abs(this->yamlReadDouble(fs, "imu.g", this->imu.g));
        this->imu.bg = this->yamlReadVec3(fs, "imu.bg", {0.0, 0.0, 0.0});
        this->imu.ba = this->yamlReadVec3(fs, "imu.ba", {0.0, 0.0, 0.0});

        this->imu.allanaN = this->yamlReadVec3(fs, "imu.allanaN", {0.0, 0.0, 0.0});
        this->imu.allanaK = this->yamlReadVec3(fs, "imu.allanaK", {0.0, 0.0, 0.0});
        this->imu.allangN = this->yamlReadVec3(fs, "imu.allangN", {0.0, 0.0, 0.0});
        this->imu.allangK = this->yamlReadVec3(fs, "imu.allangK", {0.0, 0.0, 0.0});



        fs["feat.mindist"]         >> this->vio.feat_mindist;
        fs["feat.quat"]            >> this->vio.feat_quat;
        fs["feat.bsizex"]          >> this->vio.feat_bsizex;
        fs["feat.bsizey"]          >> this->vio.feat_bsizey;
        fs["feat.max"]             >> this->vio.feat_max;
        fs["vio.rsac_it"]          >> this->vio.rsac_it;
        fs["vio.sampson_threserr"] >> this->vio.sampson_threserr;
        fs["vio.sampson_algerr"]   >> this->vio.sampson_algerr;
        fs["vio.good_para"]        >> this->vio.good_para;
        fs["track.maxlength"]      >> this->vio.track_maxlength;
        fs["track.minlength"]      >> this->vio.track_minlength;
        fs["vio.slam_pts"]         >> this->vio.slam_pts;
        fs["vio.ang_ths"]          >> this->vio.ang_ths;
        fs["vio.dis_ths"]          >> this->vio.dis_ths;

        this->vio.track_en_equalizer = this->yamlReadBool(fs, "track.en_equalizer", true);
        this->vio.track_filter_on    = this->yamlReadBool(fs, "track.filter_on", false);
        this->vio.track_show         = this->yamlReadBool(fs, "track.show", false);
        this->vio.track_shownew      = this->yamlReadBool(fs, "track.shownew", false);
        this->vio.sampson_on         = this->yamlReadBool(fs, "vio.sampson_on", true);
        this->vio.en_align           = this->yamlReadBool(fs, "vio.en_align", false);

        this->da3.enabled             = this->yamlReadBool(fs, "da3.enabled", this->da3.enabled);
        this->da3.show_window         = this->yamlReadBool(fs, "da3.show_window", this->da3.show_window);
        this->da3.model               = this->yamlReadDa3Model(fs, "da3.model", this->da3.model);
        this->da3.model_path          = this->yamlReadString(fs, "da3.model_path", defaultDa3ModelPath(this->da3.model));
        this->da3.input_width         = this->yamlReadInt(fs, "da3.input_width", this->da3.input_width);
        this->da3.input_height        = this->yamlReadInt(fs, "da3.input_height", this->da3.input_height);
        this->da3.min_valid_depth     = this->yamlReadDouble(fs, "da3.min_valid_depth", this->da3.min_valid_depth);
        this->da3.max_valid_depth     = this->yamlReadDouble(fs, "da3.max_valid_depth", this->da3.max_valid_depth);
        this->da3.evade_distance      = this->yamlReadDouble(fs, "da3.evade_distance", this->da3.evade_distance);
        this->da3.direction_deadband  = this->yamlReadDouble(fs, "da3.direction_deadband", this->da3.direction_deadband);
        this->da3.smooth_alpha        = this->yamlReadDouble(fs, "da3.smooth_alpha", this->da3.smooth_alpha);
        this->da3.debug_arrow_gain    = this->yamlReadDouble(fs, "da3.debug_arrow_gain", this->da3.debug_arrow_gain);
        this->da3.sector_count        = this->yamlReadInt(fs, "da3.sector_count", this->da3.sector_count);
        this->da3.morph_kernel_size   = this->yamlReadInt(fs, "da3.morph_kernel_size", this->da3.morph_kernel_size);
        this->da3.frontal_roi_width_ratio  = this->yamlReadDouble(fs, "da3.frontal_roi_width_ratio", this->da3.frontal_roi_width_ratio);
        this->da3.frontal_roi_height_ratio = this->yamlReadDouble(fs, "da3.frontal_roi_height_ratio", this->da3.frontal_roi_height_ratio);
        this->da3.guidance_roi_width_ratio  = this->yamlReadDouble(fs, "da3.guidance_roi_width_ratio", this->da3.guidance_roi_width_ratio);
        this->da3.guidance_roi_height_ratio = this->yamlReadDouble(fs, "da3.guidance_roi_height_ratio", this->da3.guidance_roi_height_ratio);
        this->da3.close_mask_threshold      = this->yamlReadDouble(fs, "da3.close_mask_threshold", this->da3.close_mask_threshold);
        this->da3.activate_score_threshold  = this->yamlReadDouble(fs, "da3.activate_score_threshold", this->da3.activate_score_threshold);
        this->da3.clear_score_threshold     = this->yamlReadDouble(fs, "da3.clear_score_threshold", this->da3.clear_score_threshold);
        this->da3.min_close_area_ratio      = this->yamlReadDouble(fs, "da3.min_close_area_ratio", this->da3.min_close_area_ratio);
        this->da3.min_blob_area_ratio       = this->yamlReadDouble(fs, "da3.min_blob_area_ratio", this->da3.min_blob_area_ratio);
        this->da3.emergency_close_area_ratio = this->yamlReadDouble(fs, "da3.emergency_close_area_ratio", this->da3.emergency_close_area_ratio);
        this->da3.emergency_blob_area_ratio  = this->yamlReadDouble(fs, "da3.emergency_blob_area_ratio", this->da3.emergency_blob_area_ratio);
        this->da3.score_weight_mean         = this->yamlReadDouble(fs, "da3.score_weight_mean", this->da3.score_weight_mean);
        this->da3.score_weight_area         = this->yamlReadDouble(fs, "da3.score_weight_area", this->da3.score_weight_area);
        this->da3.score_weight_blob         = this->yamlReadDouble(fs, "da3.score_weight_blob", this->da3.score_weight_blob);
        this->da3.score_weight_p20          = this->yamlReadDouble(fs, "da3.score_weight_p20", this->da3.score_weight_p20);
        this->da3.score_weight_peak         = this->yamlReadDouble(fs, "da3.score_weight_peak", this->da3.score_weight_peak);
        this->da3.frontal_peak_percentile   = this->yamlReadDouble(fs, "da3.frontal_peak_percentile", this->da3.frontal_peak_percentile);
        this->da3.peak_score_threshold      = this->yamlReadDouble(fs, "da3.peak_score_threshold", this->da3.peak_score_threshold);
        this->da3.sector_close_penalty      = this->yamlReadDouble(fs, "da3.sector_close_penalty", this->da3.sector_close_penalty);
        this->da3.free_space_power          = this->yamlReadDouble(fs, "da3.free_space_power", this->da3.free_space_power);

        if (!fs["imu.tocolor"].empty()) {
            fs["imu.tocolor"] >> this->imu.T_ci;
            this->imu.T_ci.convertTo(this->imu.T_ci, CV_64F);
        }

        return true;
    }

private:
    std::string toLower(std::string s){
        std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
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
    double yamlReadDouble(const cv::FileStorage& fs, const char * key, double fallback){
        cv::FileNode node = fs[key];
        if (node.empty()) {
            return fallback;
        }

        if (node.isReal() || node.isInt()) {
            return static_cast<double>(node);
        }

        if (node.isString()) {
            std::string temp;
            node >> temp;
            try {
                return std::stod(temp);
            } catch (...) {
                return fallback;
            }
        }

        return fallback;
    }
    int yamlReadInt(const cv::FileStorage& fs, const char * key, int fallback){
        return static_cast<int>(std::lround(yamlReadDouble(fs, key, static_cast<double>(fallback))));
    }
    std::string yamlReadString(const cv::FileStorage& fs, const char * key, const std::string& fallback){
        cv::FileNode node = fs[key];
        if (node.empty()) {
            return fallback;
        }

        if (node.isString()) {
            std::string temp;
            node >> temp;
            return temp.empty() ? fallback : temp;
        }

        return fallback;
    }
    Da3Model yamlReadDa3Model(const cv::FileStorage& fs, const char * key, Da3Model fallback){
        const std::string model = toLower(yamlReadString(fs, key, ""));
        if (model.empty()) {
            return fallback;
        }
        if (model == "small" || model == "da3_small") {
            return DA3_MODEL_SMALL;
        }
        if (model == "base" || model == "da3_base") {
            return DA3_MODEL_BASE;
        }
        return fallback;
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
    static const char* da3ModelToString(Da3Model model){
        switch(model){
            case DA3_MODEL_SMALL: return "DA3_MODEL_SMALL";
            case DA3_MODEL_BASE:  return "DA3_MODEL_BASE";
            default:              return "DA3_MODEL_UNKNOWN";
        }
    }
    static std::string defaultDa3ModelPath(Da3Model model){
        switch(model){
            case DA3_MODEL_BASE:
                return "../../model/DA3METRIC-BASE.onnx";
            case DA3_MODEL_SMALL:
            default:
                return "../../model/DA3METRIC-SMALL.onnx";
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
        os << std::fixed << std::setprecision(3) << "[" << v.x() << "," << v.y() << "," << v.z() << "]";
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
    void printVerbose() const {
        print();

        std::cout << std::fixed << std::setprecision(6);

        std::cout << "imu verbose samples:\n";

        for (size_t i = 0; i < imu.size(); ++i) {
            const ImuSample& s = imu[i];

            std::cout << "  [" << i << "] "
                    << "ts=" << s.ts
                    << " dt=" << s.dt
                    << " gyro=" << VecToStr(s.vgyr)
                    << " accel=" << VecToStr(s.vacc)
                    << "\n";
        }
    }
};

struct Pose {
    vec3 pos = vec3::Zero();
    quat rot = quat::Identity();
    vec3 getRpy() {

        const mat3 R = rot.toRotationMatrix();
        mat3 B;
        B << 0.0, -1.0,  0.0,
            0.0,  0.0, -1.0,
            1.0,  0.0,  0.0;

        const mat3 Rv = B.transpose() * R * B;
        const double roll = std::atan2(Rv(2, 1), Rv(2, 2));
        const double pitch_v = std::asin(-std::max(-1.0, std::min(1.0, Rv(2, 0))));
        const double yaw = std::atan2(Rv(1, 0), Rv(0, 0));
        return vec3(roll, -pitch_v, yaw);

    }

};

struct State {
    Pose pose;          // Pos + Ori
    ImuSample dpose;    // AngVel + LinAcc
    vec3 vel;           // Linvel  
};

struct DebugState {
    // Visual debug pose/state
    Pose vis;
    Pose imu;

    // Local preintegrated IMU delta used by the visual updater.
    vec10 preimu;       // [dp, dv, dq]
    bool imu_stat;

    // RAW IMU
    ImuSample rawimu;       // Raw Imu
    ImuSample corimu;       // Gravity, Bias and Allan corrected

    // VIO State
    uint64_t vio_inl;
    bool vio_valid;
};

struct EvitationDir {
    vec3 norm_vec = vec3::Zero();
    float magnitude = 0.0f;
    bool must_evade = false;
    float obstacle_score = 0.0f;
    float mean_closeness = 0.0f;
    float close_area_ratio = 0.0f;
    float largest_blob_ratio = 0.0f;
    float p20_closeness = 0.0f;
    float peak_closeness = 0.0f;
    float valid_ratio = 0.0f;
    float depth_p10 = 0.0f;
    float depth_p90 = 0.0f;
    float frontal_p20_depth = 0.0f;
    float frontal_peak_depth = 0.0f;
    float free_space_score = 0.0f;
    int best_sector = -1;

    double angleRad() const {
        return std::atan2(norm_vec.y(), norm_vec.x());
    }
};

struct StateOut{
    double ts_ms = 0.0;         // Time
    double dt = 0.0;
    Pose gtpose;    
    DebugState deb;             // Debug metrics
    
    Eigen::Matrix<double,15,27> H;
    vec16 x;
    State state;
    vec3 gv = vec3::Zero();
    // State vector
    Eigen::VectorXd Localx; // []
    // Information factor
    Eigen::MatrixXd LocalFactor;
    EvitationDir da3;
    

    std::vector<double> toVector(bool include_debug = false){
        std::vector<double> out;

        out.push_back(this->ts_ms);
        out.push_back(this->dt);

        out.push_back(this->deb.vis.pos.x());
        out.push_back(this->deb.vis.pos.y());
        out.push_back(this->deb.vis.pos.z());
        out.push_back(this->deb.vis.getRpy().x());
        out.push_back(this->deb.vis.getRpy().y());
        out.push_back(this->deb.vis.getRpy().z());

        out.push_back(this->deb.imu.pos.x());
        out.push_back(this->deb.imu.pos.y());
        out.push_back(this->deb.imu.pos.z());
        out.push_back(this->deb.imu.getRpy().x());
        out.push_back(this->deb.imu.getRpy().y());
        out.push_back(this->deb.imu.getRpy().z());

        out.push_back(this->gtpose.pos.x());
        out.push_back(this->gtpose.pos.y());
        out.push_back(this->gtpose.pos.z());
        out.push_back(this->gtpose.getRpy().x());
        out.push_back(this->gtpose.getRpy().y());
        out.push_back(this->gtpose.getRpy().z());

        out.push_back(this->state.pose.pos.x());
        out.push_back(this->state.pose.pos.y());
        out.push_back(this->state.pose.pos.z());
        out.push_back(this->state.pose.getRpy().x());
        out.push_back(this->state.pose.getRpy().y());
        out.push_back(this->state.pose.getRpy().z());

        out.push_back(this->state.dpose.vacc.x());
        out.push_back(this->state.dpose.vacc.y());
        out.push_back(this->state.dpose.vacc.z());
        out.push_back(this->state.dpose.vgyr.x());
        out.push_back(this->state.dpose.vgyr.y());
        out.push_back(this->state.dpose.vgyr.z());

        out.push_back(static_cast<double>(this->deb.vio_inl));
        out.push_back(static_cast<double>(this->deb.imu_stat));
        out.push_back(static_cast<double>(this->deb.vio_valid));
        out.push_back(this->da3.norm_vec.x());
        out.push_back(this->da3.norm_vec.y());
        out.push_back(this->da3.norm_vec.z());
        out.push_back(this->da3.angleRad());
        out.push_back(static_cast<double>(this->da3.magnitude));
        out.push_back(static_cast<double>(this->da3.must_evade));
        out.push_back(static_cast<double>(this->da3.obstacle_score));
        out.push_back(static_cast<double>(this->da3.mean_closeness));
        out.push_back(static_cast<double>(this->da3.close_area_ratio));
        out.push_back(static_cast<double>(this->da3.largest_blob_ratio));
        out.push_back(static_cast<double>(this->da3.p20_closeness));
        out.push_back(static_cast<double>(this->da3.peak_closeness));
        out.push_back(static_cast<double>(this->da3.valid_ratio));
        out.push_back(static_cast<double>(this->da3.depth_p10));
        out.push_back(static_cast<double>(this->da3.depth_p90));
        out.push_back(static_cast<double>(this->da3.frontal_p20_depth));
        out.push_back(static_cast<double>(this->da3.frontal_peak_depth));
        out.push_back(static_cast<double>(this->da3.free_space_score));
        out.push_back(static_cast<double>(this->da3.best_sector));

        for(int i = 0; i < this->deb.preimu.size(); ++i){
            out.push_back(*(this->deb.preimu.data() + i));
        }
        return out;
    }
};


struct Command{
    double ts_ms;
    vec3 lenvel_ms;
    vec3 angvel_rads;
};

struct Waypoints{
    std::vector<vec3> p;
};