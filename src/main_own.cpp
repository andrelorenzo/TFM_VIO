#define LOGGER_IMP
#include "seconds/logger.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "CSVlogger.hpp"
#include "config.hpp"
#include "imu_vio.hpp"
#include "source_man.hpp"
#include "visual_odometry.hpp"
#include "gt_estimator.hpp"

namespace {

static bool readVec3Node(const cv::FileStorage& fs, const std::string& key, cv::Vec3d* out)
{
    if (out == nullptr) {
        return false;
    }

    const cv::FileNode node = fs[key];
    if (node.empty()) {
        std::cerr << "Clave YAML ausente: " << key << std::endl;
        return false;
    }

    cv::Mat tmp;
    node >> tmp;
    if (tmp.empty()) {
        std::cerr << "Nodo YAML vacio en: " << key << std::endl;
        return false;
    }

    tmp = tmp.reshape(1, 1);
    tmp.convertTo(tmp, CV_64F);
    if (tmp.total() != 3) {
        std::cerr << "Nodo YAML invalido (esperado vector de 3) en: " << key << std::endl;
        return false;
    }

    *out = cv::Vec3d(tmp.at<double>(0, 0), tmp.at<double>(0, 1), tmp.at<double>(0, 2));
    return true;
}

static std::string trimCopy(const std::string& s)
{
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

static std::string lowerCopy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}
static bool readBoolNode(const cv::FileStorage& fs, const std::string& key, bool* out)
{
    if (out == nullptr) {
        return false;
    }

    const cv::FileNode node = fs[key];
    if (node.empty()) {
        std::cerr << "Clave YAML ausente: " << key << std::endl;
        return false;
    }

    if (node.isInt()) {
        *out = (static_cast<int>(node) != 0);
        return true;
    }
    if (node.isReal()) {
        *out = (std::abs(static_cast<double>(node)) > 0.5);
        return true;
    }
    if (node.isString()) {
        const std::string s = lowerCopy(trimCopy(static_cast<std::string>(node)));
        if (s == "true" || s == "1" || s == "yes" || s == "on") {
            *out = true;
            return true;
        }
        if (s == "false" || s == "0" || s == "no" || s == "off") {
            *out = false;
            return true;
        }
        std::cerr << "Valor booleano invalido en: " << key << " -> \"" << s << "\"" << std::endl;
        return false;
    }

    std::cerr << "Nodo YAML invalido para bool en: " << key << std::endl;
    return false;
}
static bool readMat33NodeOptional(const cv::FileStorage& fs,const std::vector<std::string>& keys,cv::Matx33d* out){
    if (out == nullptr) {
        return false;
    }

    for (const auto& key : keys) {
        const cv::FileNode node = fs[key];
        if (node.empty()) {
            continue;
        }

        cv::Mat tmp;
        node >> tmp;
        if (tmp.empty()) {
            continue;
        }

        tmp.convertTo(tmp, CV_64F);
        if (tmp.rows == 3 && tmp.cols == 3) {
            *out = cv::Matx33d(
                tmp.at<double>(0, 0), tmp.at<double>(0, 1), tmp.at<double>(0, 2),
                tmp.at<double>(1, 0), tmp.at<double>(1, 1), tmp.at<double>(1, 2),
                tmp.at<double>(2, 0), tmp.at<double>(2, 1), tmp.at<double>(2, 2));
            return true;
        }
    }

    return false;
}
bool parseConfig(const char* path, Config* config){
    if (config == nullptr) {
        std::cerr << "Config nulo" << std::endl;
        return false;
    }

    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "No se pudo abrir el YAML: " << path << std::endl;
        return false;
    }

    if (!readBoolNode(fs, "gen.show", &config->gen.show)) return false;
    if (!readBoolNode(fs, "gen.mono-inertial", &config->gen.mono_inertial)) return false;
    fs["gen.input"] >> config->gen.input;
    fs["gen.out"] >> config->gen.output;
    fs["gen.script"] >> config->gen.script;
    if (!readBoolNode(fs, "gen.gt", &config->gen.calc_gt)) return false;
    if (!readBoolNode(fs, "gen.debug", &config->gen.debug)) return false;
    if (!readBoolNode(fs, "gen.use_allan", &config->gen.use_allan)) return false;

    const std::string input_lc = lowerCopy(config->gen.input);
    if (config->gen.input.empty()) {
        config->gen.in_type = FEED_RSCAM;
    } else if (input_lc.find(".bag") != std::string::npos) {
        config->gen.in_type = FEED_BAG;
    } else if (input_lc.find(".csv") != std::string::npos) {
        config->gen.in_type = FEED_CSV;
    } else if (input_lc.find("rtsp://") != std::string::npos) {
        config->gen.in_type = FEED_RTSP;
    } else if (input_lc == "cam") {
        config->gen.in_type = FEED_PCCAM;
    } else if (input_lc.find("com") != std::string::npos || input_lc.find("/dev/tty") != std::string::npos) {
        config->gen.in_type = FEED_PORT;
    } else if (input_lc.find(".mp4") != std::string::npos) {
        config->gen.in_type = FEED_MP4;
    } else {
        config->gen.in_type = FEED_RSCAM;
    }

    if (!readBoolNode(fs, "cam.imu", &config->cam.has_imu)) return false;
    fs["cam.width"] >> config->cam.w;
    fs["cam.height"] >> config->cam.h;
    fs["cam.fps"] >> config->cam.fps;
    fs["cam.RGB"] >> config->cam.rgb_format;

    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
    fs["cam.fx"] >> fx;
    fs["cam.fy"] >> fy;
    fs["cam.cx"] >> cx;
    fs["cam.cy"] >> cy;
    config->cam.K = (cv::Mat_<double>(3, 3) <<
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0);

    fs["cam.dist"] >> config->cam.D;
    config->cam.D.convertTo(config->cam.D, CV_64F);

    if (!readVec3Node(fs, "imu.gv", &config->imu.gv)) return false;
    if (!readVec3Node(fs, "imu.bg", &config->imu.gyro_bias)) return false;
    if (!readVec3Node(fs, "imu.ba", &config->imu.accel_bias)) return false;
    if (!readVec3Node(fs, "imu.allangx", &config->imu.allan_gx_nbk)) return false;
    if (!readVec3Node(fs, "imu.allangy", &config->imu.allan_gy_nbk)) return false;
    if (!readVec3Node(fs, "imu.allangz", &config->imu.allan_gz_nbk)) return false;
    if (!readVec3Node(fs, "imu.allanax", &config->imu.allan_ax_nbk)) return false;
    if (!readVec3Node(fs, "imu.allanay", &config->imu.allan_ay_nbk)) return false;
    if (!readVec3Node(fs, "imu.allanaz", &config->imu.allan_az_nbk)) return false;

    readMat33NodeOptional(fs,
                          {"imu.gyro_scale_misalignment", "imu.gyro_sm", "imu.gsm"},
                          &config->imu.gyro_scale_misalignment);
    readMat33NodeOptional(fs,
                          {"imu.accel_scale_misalignment", "imu.accel_sm", "imu.asm"},
                          &config->imu.accel_scale_misalignment);

    fs["imu.freq"] >> config->imu.freq;
    if (!fs["imu.tocolor"].empty()) {
        fs["imu.tocolor"] >> config->imu.T;
        config->imu.T.convertTo(config->imu.T, CV_64F);
    }

    fs["orb.nFeatures"] >> config->orb.nFeatures;
    fs["orb.scaleFactor"] >> config->orb.scaleFactor;
    fs["orb.nLevels"] >> config->orb.nLevels;
    fs["orb.iniThFAST"] >> config->orb.iniThFAST;
    fs["orb.minThFAST"] >> config->orb.minThFAST;

    if (config->imu.freq <= 0.0) {
        std::cerr << "Frecuencia de IMU invalida" << std::endl;
        return false;
    }
    if (!config->imu.hasValidGravity()) {
        std::cerr << "Vector de gravedad imu.gv invalido" << std::endl;
        return false;
    }

    if (config->gen.in_type != FEED_CSV && config->gen.in_type != FEED_PORT) {
        if (config->cam.K.empty() || config->cam.K.rows != 3 || config->cam.K.cols != 3) {
            std::cerr << "Matriz intrinseca K invalida" << std::endl;
            return false;
        }
        if (config->cam.D.empty()) {
            std::cerr << "Matriz de distorsion vacia" << std::endl;
            return false;
        }
    }

    return true;
}
void launchScript(const char* script){
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const int ret = std::system(script);
    if (ret != 0) {
        Logger(ERROR, "Python script returned error code %d", ret);
    }
}

cv::Mat toGray(const cv::Mat& color, int rgb_format)
{
    if (color.empty()) {
        return cv::Mat();
    }
    if (color.channels() == 1) {
        return color.clone();
    }

    cv::Mat gray;
    const int code = (rgb_format == 1) ? cv::COLOR_RGB2GRAY : cv::COLOR_BGR2GRAY;
    cv::cvtColor(color, gray, code);
    return gray;
}

cv::Mat toDisplayBgr(const cv::Mat& color, int rgb_format)
{
    if (color.empty()) {
        return cv::Mat();
    }
    if (color.channels() == 1) {
        cv::Mat out;
        cv::cvtColor(color, out, cv::COLOR_GRAY2BGR);
        return out;
    }
    if (rgb_format == 1) {
        cv::Mat out;
        cv::cvtColor(color, out, cv::COLOR_RGB2BGR);
        return out;
    }
    return color.clone();
}

void overlayImuInfo(const ImuFrontendOutput& imu,
                    const vo::VioPoseOutput& vio,
                    bool have_gt,
                    const cv::Vec3d& gt_xyz,
                    cv::Mat* vis)
{
    if (vis == nullptr || vis->empty()) {
        return;
    }

    std::ostringstream oss1, oss2, oss3, oss4, oss5;
    oss1 << std::fixed << std::setprecision(3)
         << "IMU pos = [" << imu.pos_xyz[0] << ", " << imu.pos_xyz[1] << ", " << imu.pos_xyz[2] << "] m";
    oss2 << std::fixed << std::setprecision(3)
         << "IMU vel = [" << imu.vel_xyz[0] << ", " << imu.vel_xyz[1] << ", " << imu.vel_xyz[2] << "] m/s";
    oss3 << std::fixed << std::setprecision(2)
         << "IMU rpy = [" << imu.rpy_deg[0] << ", " << imu.rpy_deg[1] << ", " << imu.rpy_deg[2] << "] deg"
         << (imu.stationary ? "  stationary" : "");
    oss4 << std::fixed << std::setprecision(3)
         << "Preint dt=" << imu.preint_dt
         << "  dp=[" << imu.preint_dp[0] << ", " << imu.preint_dp[1] << ", " << imu.preint_dp[2] << "]";
    oss5 << std::fixed << std::setprecision(3)
         << "VIO scale=" << vio.scale_estimate
         << (vio.scale_initialized ? "" : " (warmup)");

    cv::putText(*vis, oss1.str(), cv::Point(12, 168), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(*vis, oss2.str(), cv::Point(12, 192), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(*vis, oss3.str(), cv::Point(12, 216), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
    cv::putText(*vis, oss4.str(), cv::Point(12, 240), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(255, 220, 0), 2, cv::LINE_AA);
    cv::putText(*vis, oss5.str(), cv::Point(12, 264), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    if (have_gt) {
        std::ostringstream oss_gt;
        oss_gt << std::fixed << std::setprecision(3)
               << "GT xyz = [" << gt_xyz[0] << ", " << gt_xyz[1] << ", " << gt_xyz[2] << "]";
        cv::putText(*vis, oss_gt.str(), cv::Point(12, 288), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
}

std::vector<std::string> makeCsvHeader()
{
    return {
        "timestamp",
        "frame_timestamp",
        "imu_timestamp",
        "frame_number",
        "imu_initialized",
        "imu_stationary",
        "vio_initialized",
        "vio_valid",
        "vio_imu_used",
        "scale_initialized",
        "scale_updated",
        "scale_estimate",
        "fps",
        "raw_matches",
        "tracked_points",
        "pose_inliers",
        "median_flow_px",
        "x", "y", "z",
        "roll_deg", "pitch_deg", "yaw_deg",
        "quat_w", "quat_x", "quat_y", "quat_z",
        "x_visual_unscaled", "y_visual_unscaled", "z_visual_unscaled",
        "roll_deg_visual", "pitch_deg_visual", "yaw_deg_visual",
        "x_imu", "y_imu", "z_imu",
        "roll_deg_imu", "pitch_deg_imu", "yaw_deg_imu",
        "imu_pos_x", "imu_pos_y", "imu_pos_z",
        "imu_vel_x", "imu_vel_y", "imu_vel_z",
        "imu_roll_deg", "imu_pitch_deg", "imu_yaw_deg",
        "preint_dt",
        "preint_dtheta_x", "preint_dtheta_y", "preint_dtheta_z",
        "preint_dv_x", "preint_dv_y", "preint_dv_z",
        "preint_dp_x", "preint_dp_y", "preint_dp_z",
        "gt_x", "gt_y", "gt_z",
        "gyro_raw_x","gyro_raw_y","gyro_raw_z",
        "gyro_cal_x","gyro_cal_y","gyro_cal_z",
        "acc_raw_x","acc_raw_y","acc_raw_z",
        "acc_cal_x","acc_cal_y","acc_cal_z"
    };
}

std::vector<double> makeCsvRow(const vo::VioPoseOutput* vio,const ImuFrontendOutput* imu,double frame_ts,std::uint64_t frame_number,const cv::Vec3d& gt_xyz){
    const double nan = std::numeric_limits<double>::quiet_NaN();

    const bool have_vio = (vio != nullptr);
    const bool have_imu = (imu != nullptr);

    const double timestamp = have_vio ? vio->timestamp_sec : (have_imu ? imu->timestamp_sec : 0.0);
    const double imu_ts = have_imu ? imu->timestamp_sec : nan;

    return {
        timestamp,
        frame_ts,
        imu_ts,
        static_cast<double>(frame_number),
        have_imu ? (imu->initialized ? 1.0 : 0.0) : 0.0,
        have_imu ? (imu->stationary ? 1.0 : 0.0) : 0.0,
        have_vio ? (vio->initialized ? 1.0 : 0.0) : 0.0,
        have_vio ? (vio->valid ? 1.0 : 0.0) : 0.0,
        have_vio ? (vio->imu_used ? 1.0 : 0.0) : 0.0,
        have_vio ? (vio->scale_initialized ? 1.0 : 0.0) : 0.0,
        have_vio ? (vio->scale_updated ? 1.0 : 0.0) : 0.0,
        have_vio ? vio->scale_estimate : nan,
        have_vio ? vio->fps : nan,
        have_vio ? static_cast<double>(vio->raw_matches) : nan,
        have_vio ? static_cast<double>(vio->tracked_points) : nan,
        have_vio ? static_cast<double>(vio->pose_inliers) : nan,
        have_vio ? vio->median_flow_px : nan,
        have_vio ? vio->x : nan,
        have_vio ? vio->y : nan,
        have_vio ? vio->z : nan,
        have_vio ? vio->roll_deg : nan,
        have_vio ? vio->pitch_deg : nan,
        have_vio ? vio->yaw_deg : nan,
        have_vio ? vio->quat_wxyz[0] : nan,
        have_vio ? vio->quat_wxyz[1] : nan,
        have_vio ? vio->quat_wxyz[2] : nan,
        have_vio ? vio->quat_wxyz[3] : nan,
        have_vio ? vio->x_visual_unscaled : nan,
        have_vio ? vio->y_visual_unscaled : nan,
        have_vio ? vio->z_visual_unscaled : nan,
        have_vio ? vio->roll_deg_visual : nan,
        have_vio ? vio->pitch_deg_visual : nan,
        have_vio ? vio->yaw_deg_visual : nan,
        have_vio ? vio->x_imu : nan,
        have_vio ? vio->y_imu : nan,
        have_vio ? vio->z_imu : nan,
        have_vio ? vio->roll_deg_imu : nan,
        have_vio ? vio->pitch_deg_imu : nan,
        have_vio ? vio->yaw_deg_imu : nan,
        have_imu ? imu->pos_xyz[0] : nan,
        have_imu ? imu->pos_xyz[1] : nan,
        have_imu ? imu->pos_xyz[2] : nan,
        have_imu ? imu->vel_xyz[0] : nan,
        have_imu ? imu->vel_xyz[1] : nan,
        have_imu ? imu->vel_xyz[2] : nan,
        have_imu ? imu->rpy_deg[0] : nan,
        have_imu ? imu->rpy_deg[1] : nan,
        have_imu ? imu->rpy_deg[2] : nan,
        have_imu ? imu->preint_dt : nan,
        have_imu ? imu->preint_dtheta[0] : nan,
        have_imu ? imu->preint_dtheta[1] : nan,
        have_imu ? imu->preint_dtheta[2] : nan,
        have_imu ? imu->preint_dv[0] : nan,
        have_imu ? imu->preint_dv[1] : nan,
        have_imu ? imu->preint_dv[2] : nan,
        have_imu ? imu->preint_dp[0] : nan,
        have_imu ? imu->preint_dp[1] : nan,
        have_imu ? imu->preint_dp[2] : nan,
        gt_xyz[0], gt_xyz[1], gt_xyz[2],
        have_imu ? imu->gyro_raw[0] : nan,
        have_imu ? imu->gyro_raw[1] : nan,
        have_imu ? imu->gyro_raw[2] : nan,
        have_imu ? imu->gyro_cal[0] : nan,
        have_imu ? imu->gyro_cal[1] : nan,
        have_imu ? imu->gyro_cal[2] : nan,
        have_imu ? imu->accel_raw[0] : nan,
        have_imu ? imu->accel_raw[1] : nan,
        have_imu ? imu->accel_raw[2] : nan,
        have_imu ? imu->accel_cal[0] : nan,
        have_imu ? imu->accel_cal[1] : nan,
        have_imu ? imu->accel_cal[2] : nan
    };
}

} // namespace

int main(int argc, char** argv)
{
    LoggerSetVerbsity(DEBUG);

    if (argc < 2) {
        Logger(ERROR, "Usage: %s <path-to-config.yaml>", argv[0]);
        return -1;
    }

    Config config;
    if (!parseConfig(argv[1], &config)) {
        Logger(ERROR, "Error while parsing config YAML");
        return -1;
    }

    config.print();

    if (!initSourceManager(config)) {
        Logger(ERROR, "Error initializing source manager");
        return -1;
    }

    ImuFrontend imu_frontend;
    bool imu_enabled = config.cam.has_imu;
    if (imu_enabled) {
        try {
            imu_frontend.init(config);
            Logger(INFO,
                   "Frontend IMU inicializado (%s Allan, bias fijo del config, preintegracion por frame activa)",
                   (config.gen.use_allan && config.imu.hasValidAllan()) ? "con" : "sin");
        } catch (const std::exception& e) {
            Logger(ERROR, "Error inicializando frontend IMU: %s", e.what());
            closeSourceManager();
            return -1;
        }
    }

    const bool visual_enabled = (config.gen.in_type != FEED_CSV && config.gen.in_type != FEED_PORT);
    std::unique_ptr<vo::VisualInertialOdometry> vio_system;
    if (visual_enabled) {
        try {
            vio_system.reset(new vo::VisualInertialOdometry(config));
            Logger(INFO,
                   "Sistema visual%s inicializado.",
                   (config.gen.mono_inertial && imu_enabled) ? "-inercial" : "");
        } catch (const std::exception& e) {
            Logger(ERROR, "Error inicializando VisualInertialOdometry: %s", e.what());
            closeSourceManager();
            return -1;
        }
    }

    GroundTruthEstimator gt;
    GroundTruthConfig gt_cfg;
    GroundTruthFrame gt_frame;
    bool gt_enabled = config.gen.calc_gt;
    GroundTruthState gt_state;
    if (gt_enabled) {
        gt_cfg.enabled = true;
        if (!gt.init(gt_cfg)) {
            Logger(WARN, "No se pudo inicializar GroundTruthEstimator; se desactiva GT.");
            gt_enabled = false;
        }
    }

    if (config.gen.show && visual_enabled) {
        cv::namedWindow("vio", cv::WINDOW_NORMAL);
        cv::resizeWindow("vio", 1280, 720);
    }

    CsvLogger logger;
    if (!logger.init(config.gen.output.c_str(), makeCsvHeader())) {
        Logger(ERROR, "Error creating csv file: %s", config.gen.output.c_str());
        closeSourceManager();
        cv::destroyAllWindows();
        return -1;
    }

    if (!config.gen.script.empty()) {
        std::thread(launchScript, config.gen.script.c_str()).detach();
    }

    sourcePacket packet;
    ImuFrontendOutput last_imu_state;
    bool have_imu_state = false;
    vo::VioPoseOutput last_vio_state;
    bool have_vio_state = false;
    cv::Vec3d gt_xyz(0.0, 0.0, 0.0);

    while (true) {
        if (!getSourceManager(&packet)) {
            Logger(INFO, "No more data from source");
            break;
        }

        if (packet.imu_data.empty() && !packet.new_color && !packet.new_depth) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        bool fatal_error = false;

        for (const auto& imu_sample : packet.imu_data) {
            if (!imu_enabled) {
                break;
            }
            try {
                static double last_imu_ts_processed = -std::numeric_limits<double>::infinity();

                if (!(imu_sample.ts > last_imu_ts_processed + 1e-9)) {
                    Logger(WARN, "Skipping non-increasing IMU sample ts=%.9f last=%.9f",
                        imu_sample.ts, last_imu_ts_processed);
                    continue;
                }

                last_imu_ts_processed = imu_sample.ts;
                last_imu_state = imu_frontend.update(imu_sample);
                have_imu_state = true;
            } catch (const std::exception& e) {
                Logger(ERROR, "Error actualizando frontend IMU: %s", e.what());
                fatal_error = true;
                break;
            }
        }
        if (fatal_error) {
            break;
        }

        if (gt_enabled && packet.new_color && packet.new_depth && !packet.color.empty() && !packet.depth.empty()) {
            try {
                gt_frame.rgb = toDisplayBgr(packet.color, config.cam.rgb_format);
                gt_frame.gray = toGray(packet.color, config.cam.rgb_format);
                gt_frame.depth_m = packet.depth;
                gt_frame.K = config.cam.K;
                gt_frame.dist = config.cam.D;
                gt_frame.timestamp_sec = static_cast<double>(packet.colorts_ms) * 1e-3;
                gt_frame.frame_number = packet.color_frame_number;
                gt_state = gt.update(gt_frame);
                gt_xyz = gt_state.xyz;
            } catch (const std::exception& e) {
                Logger(WARN, "Ground truth update skipped: %s", e.what());
            }
        }

        if (visual_enabled && packet.new_color && !packet.color.empty()) {
            cv::Mat frame_bgr = toDisplayBgr(packet.color, config.cam.rgb_format);
            cv::Mat frame_gray = toGray(packet.color, config.cam.rgb_format);

            vo::VisionFrame frame;
            frame.bgr = frame_bgr;
            frame.gray = frame_gray;
            frame.timestamp_sec = static_cast<double>(packet.colorts_ms) * 1e-3;
            frame.frame_number = packet.color_frame_number;

            const ImuFrontendOutput* imu_for_vio = nullptr;
            if (config.gen.mono_inertial && imu_enabled && have_imu_state && last_imu_state.initialized) {
                imu_for_vio = &last_imu_state;
            }

            cv::Mat debug_vis;
            try {
                last_vio_state = vio_system->update(frame,
                                                    imu_for_vio,
                                                    (config.gen.show || config.gen.debug) ? &debug_vis : nullptr);
                have_vio_state = true;
            } catch (const std::exception& e) {
                Logger(ERROR, "Error actualizando VIO: %s", e.what());
                if (imu_enabled) {
                    imu_frontend.notifyFrameBoundary();
                }
                continue;
            }

            if (imu_enabled) {
                imu_frontend.notifyFrameBoundary();
            }

            if (debug_vis.empty()) {
                debug_vis = frame_bgr.clone();
            }
            if (have_imu_state) {
                overlayImuInfo(last_imu_state, last_vio_state, gt_enabled, gt_xyz, &debug_vis);
            }

            if (config.gen.show) {
                cv::imshow("vio", debug_vis);
            }

            logger.addRow(makeCsvRow(have_vio_state ? &last_vio_state : nullptr,
                                     have_imu_state ? &last_imu_state : nullptr,
                                     frame.timestamp_sec,
                                     frame.frame_number,
                                     gt_xyz));

            if (config.gen.debug) {
                Logger(DEBUG,
                       "frame=%llu t=%.3f vio_valid=%d imu_used=%d scale=%.4f xyz=[%.3f %.3f %.3f] rpy=[%.2f %.2f %.2f] matches=%d inliers=%d",
                       static_cast<unsigned long long>(frame.frame_number),
                       frame.timestamp_sec,
                       last_vio_state.valid ? 1 : 0,
                       last_vio_state.imu_used ? 1 : 0,
                       last_vio_state.scale_estimate,
                       last_vio_state.x, last_vio_state.y, last_vio_state.z,
                       last_vio_state.roll_deg, last_vio_state.pitch_deg, last_vio_state.yaw_deg,
                       last_vio_state.raw_matches,
                       last_vio_state.pose_inliers);
            }
        } else if (!visual_enabled && have_imu_state) {
            logger.addRow(makeCsvRow(nullptr,
                                     &last_imu_state,
                                     last_imu_state.timestamp_sec,
                                     0,
                                     gt_xyz));
        }

        if (config.gen.show) {
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }
    }

    logger.close();
    closeSourceManager();
    cv::destroyAllWindows();
    return 0;
}
