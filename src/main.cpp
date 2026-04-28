#define LOGGER_IMP
#include "seconds/logger.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "CSVlogger.hpp"
#include "System.h"
#include "config.hpp"
#include "imu.hpp"
#include "imu_vio.hpp"
#include "source_man.hpp"
#include "visual_odometry.hpp"

#if __has_include("gt_estimator.hpp")
#define HAVE_GT_ESTIMATOR 1
#include "gt_estimator.hpp"
#else
#define HAVE_GT_ESTIMATOR 0
#endif

namespace {

bool readVec3Node(const cv::FileStorage& fs, const std::string& key, cv::Vec3d* out)
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

std::string trimCopy(const std::string& s)
{
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::string lowerCopy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool readBoolNode(const cv::FileStorage& fs, const std::string& key, bool* out)
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

bool readMat33NodeOptional(const cv::FileStorage& fs,
                           const std::vector<std::string>& keys,
                           cv::Matx33d* out)
{
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

bool parseConfig(const char* path, Config* config)
{
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

    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
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

void launchScript(const char* script)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const int ret = std::system(script);
    if (ret != 0) {
        Logger(ERROR, "Python script returned error code %d", ret);
    }
}

cv::Mat quatXyzwToRotation(const cv::Vec4d& q_in)
{
    double x = q_in[0];
    double y = q_in[1];
    double z = q_in[2];
    double w = q_in[3];

    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    if (n <= 1e-12 || !std::isfinite(n)) {
        return cv::Mat::eye(3, 3, CV_64F);
    }

    x /= n;
    y /= n;
    z /= n;
    w /= n;

    return (cv::Mat_<double>(3, 3)
        << 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
           2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
           2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y));
}

cv::Vec3d rotationToRpyDeg(const cv::Mat& R_wc)
{
    cv::Mat R;
    R_wc.convertTo(R, CV_64F);

    const double roll = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
    const double pitch = std::asin(-std::max(-1.0, std::min(1.0, R.at<double>(2, 0))));
    const double yaw = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    const double kRadToDeg = 180.0 / 3.14159265358979323846;
    return cv::Vec3d(roll * kRadToDeg, pitch * kRadToDeg, yaw * kRadToDeg);
}

struct RvioPoseOutput
{
    int image_id = -1;
    std::uint64_t frame_number = 0;
    double timestamp_sec = 0.0;
    cv::Vec3d position_xyz = {0.0, 0.0, 0.0};
    cv::Vec4d quat_xyzw = {0.0, 0.0, 0.0, 1.0};
    cv::Vec3d rpy_deg = {0.0, 0.0, 0.0};
};

enum class ExecutionMode
{
    Rvio2Vio,
    VisualOnly,
    ImuOnly,
    Passive
};

const char* executionModeName(ExecutionMode mode)
{
    switch (mode) {
        case ExecutionMode::Rvio2Vio:
            return "rvio2-vio";
        case ExecutionMode::VisualOnly:
            return "visual-only";
        case ExecutionMode::ImuOnly:
            return "imu-only";
        case ExecutionMode::Passive:
        default:
            return "passive";
    }
}

std::optional<RvioPoseOutput> poseFromVisualOutput(const vo::VioPoseOutput& pose,
                                                   double timestampSec,
                                                   std::uint64_t frameNumber)
{
    if (!pose.initialized) {
        return std::nullopt;
    }

    RvioPoseOutput out;
    out.image_id = pose.frame_idx;
    out.frame_number = frameNumber;
    out.timestamp_sec = timestampSec;
    out.position_xyz = cv::Vec3d(pose.x, pose.y, pose.z);
    out.quat_xyzw = cv::Vec4d(
        pose.quat_wxyz[1],
        pose.quat_wxyz[2],
        pose.quat_wxyz[3],
        pose.quat_wxyz[0]);
    out.rpy_deg = cv::Vec3d(pose.roll_deg, pose.pitch_deg, pose.yaw_deg);
    return out;
}

std::optional<RvioPoseOutput> poseFromImuOutput(const ImuFrontendOutput& imu,
                                                std::uint64_t frameNumber = 0)
{
    if (!imu.initialized) {
        return std::nullopt;
    }

    RvioPoseOutput out;
    out.image_id = -1;
    out.frame_number = frameNumber;
    out.timestamp_sec = imu.timestamp_sec;
    out.position_xyz = imu.pos_xyz;
    out.quat_xyzw = cv::Vec4d(
        imu.quat_wxyz[1],
        imu.quat_wxyz[2],
        imu.quat_wxyz[3],
        imu.quat_wxyz[0]);
    out.rpy_deg = imu.rpy_deg;
    return out;
}

class Rvio2Adapter
{
public:
    explicit Rvio2Adapter(const std::string& configPath)
        : system_(configPath)
    {
    }

    void pushImuSample(const imuData& sample)
    {
        std::scoped_lock lock(mutex_);

        RVIO2::ImuData* data = new RVIO2::ImuData();
        data->Timestamp = sample.ts;
        data->AngularVel << static_cast<float>(sample.gyro[0]),
                             static_cast<float>(sample.gyro[1]),
                             static_cast<float>(sample.gyro[2]);
        data->LinearAccel << static_cast<float>(sample.accel[0]),
                              static_cast<float>(sample.accel[1]),
                              static_cast<float>(sample.accel[2]);
        data->TimeInterval = (lastImuTimestampSec_ < 0.0) ? 0.0 : (sample.ts - lastImuTimestampSec_);

        lastImuTimestampSec_ = sample.ts;
        system_.PushImuData(data);
    }

    std::optional<RvioPoseOutput> pushImageFrame(double timestampSec,
                                                 std::uint64_t frameNumber,
                                                 const cv::Mat& image)
    {
        std::scoped_lock lock(mutex_);

        RVIO2::ImageData* data = new RVIO2::ImageData();
        data->Timestamp = timestampSec;
        data->Image = image.clone();
        system_.PushImageData(data);

        RVIO2::PoseEstimate pose;
        if (!system_.run(&pose)) {
            return std::nullopt;
        }

        return convertPose(pose, frameNumber);
    }

    bool isInitialized() const
    {
        std::scoped_lock lock(mutex_);
        return system_.IsInitialized();
    }

private:
    static RvioPoseOutput convertPose(const RVIO2::PoseEstimate& pose, std::uint64_t frameNumber)
    {
        RvioPoseOutput out;
        out.image_id = pose.ImageId;
        out.frame_number = frameNumber;
        out.timestamp_sec = pose.Timestamp;
        out.position_xyz = cv::Vec3d(pose.Position(0), pose.Position(1), pose.Position(2));
        out.quat_xyzw = cv::Vec4d(pose.Quaternion(0), pose.Quaternion(1), pose.Quaternion(2), pose.Quaternion(3));
        out.rpy_deg = rotationToRpyDeg(quatXyzwToRotation(out.quat_xyzw));
        return out;
    }

private:
    mutable std::mutex mutex_;
    double lastImuTimestampSec_ = -1.0;
    RVIO2::System system_;
};

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

void overlayPoseInfo(const std::optional<RvioPoseOutput>& pose,
                     const char* systemLabel,
                     bool systemInitialized,
                     bool have_gt,
                     const cv::Vec3d& gt_xyz,
                     cv::Mat* vis)
{
    if (vis == nullptr || vis->empty()) {
        return;
    }

    if (pose.has_value()) {
        std::ostringstream oss1;
        std::ostringstream oss2;
        oss1 << std::fixed << std::setprecision(3)
             << systemLabel << " xyz = [" << pose->position_xyz[0] << ", "
             << pose->position_xyz[1] << ", "
             << pose->position_xyz[2] << "]";
        oss2 << std::fixed << std::setprecision(2)
             << systemLabel << " rpy = [" << pose->rpy_deg[0] << ", "
             << pose->rpy_deg[1] << ", "
             << pose->rpy_deg[2] << "] deg";

        cv::putText(*vis, oss1.str(), cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        cv::putText(*vis, oss2.str(), cv::Point(12, 48), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    } else {
        const std::string status = systemInitialized
            ? (std::string(systemLabel) + " sin pose para este frame")
            : (std::string(systemLabel) + " esperando datos / inicializacion");
        cv::putText(*vis, status, cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.58,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }

    if (have_gt) {
        std::ostringstream oss_gt;
        oss_gt << std::fixed << std::setprecision(3)
               << "GT xyz = [" << gt_xyz[0] << ", " << gt_xyz[1] << ", " << gt_xyz[2] << "]";
        cv::putText(*vis, oss_gt.str(), cv::Point(12, 72), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
}

void overlayGroundTruthInfo(bool have_gt,
                            const cv::Vec3d& gt_xyz,
                            int y,
                            cv::Mat* vis)
{
    if (!have_gt || vis == nullptr || vis->empty()) {
        return;
    }

    std::ostringstream oss_gt;
    oss_gt << std::fixed << std::setprecision(3)
           << "GT xyz = [" << gt_xyz[0] << ", " << gt_xyz[1] << ", " << gt_xyz[2] << "]";
    cv::putText(*vis, oss_gt.str(), cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.52,
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

std::vector<std::string> makeCsvHeader()
{
    return {
        "timestamp",
        "pose_x", "pose_y", "pose_z",
        "roll_deg", "pitch_deg", "yaw_deg",
        "gyro_raw_x", "gyro_raw_y", "gyro_raw_z",
        "acc_raw_x", "acc_raw_y", "acc_raw_z",
        "gyro_cal_x", "gyro_cal_y", "gyro_cal_z",
        "acc_cal_x", "acc_cal_y", "acc_cal_z",
        "gt_x", "gt_y", "gt_z"
    };
}

std::vector<double> makeCsvRow(double timestamp,
                               const std::optional<RvioPoseOutput>& pose,
                               const imuData* imuRaw,
                               const imuData* imuCal,
                               const cv::Vec3d& gt_xyz)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();

    return {
        timestamp,
        pose ? pose->position_xyz[0] : nan,
        pose ? pose->position_xyz[1] : nan,
        pose ? pose->position_xyz[2] : nan,
        pose ? pose->rpy_deg[0] : nan,
        pose ? pose->rpy_deg[1] : nan,
        pose ? pose->rpy_deg[2] : nan,
        imuRaw ? imuRaw->gyro[0] : nan,
        imuRaw ? imuRaw->gyro[1] : nan,
        imuRaw ? imuRaw->gyro[2] : nan,
        imuRaw ? imuRaw->accel[0] : nan,
        imuRaw ? imuRaw->accel[1] : nan,
        imuRaw ? imuRaw->accel[2] : nan,
        imuCal ? imuCal->gyro[0] : nan,
        imuCal ? imuCal->gyro[1] : nan,
        imuCal ? imuCal->gyro[2] : nan,
        imuCal ? imuCal->accel[0] : nan,
        imuCal ? imuCal->accel[1] : nan,
        imuCal ? imuCal->accel[2] : nan,
        gt_xyz[0], gt_xyz[1], gt_xyz[2]
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

    const bool visualEnabled = (config.gen.in_type != FEED_CSV && config.gen.in_type != FEED_PORT);
    ExecutionMode mode = ExecutionMode::Passive;
    if (visualEnabled) {
        mode = (config.cam.has_imu && config.gen.mono_inertial)
            ? ExecutionMode::Rvio2Vio
            : ExecutionMode::VisualOnly;
    } else if (config.cam.has_imu) {
        mode = ExecutionMode::ImuOnly;
    }

    Logger(INFO, "Modo de ejecucion seleccionado: %s", executionModeName(mode));

    std::unique_ptr<Rvio2Adapter> rvioSystem;
    std::unique_ptr<vo::VisualInertialOdometry> visualSystem;
    std::unique_ptr<ImuFrontend> imuFrontend;
    if (mode == ExecutionMode::Rvio2Vio) {
        try {
            rvioSystem = std::make_unique<Rvio2Adapter>(argv[1]);
            Logger(INFO, "Sistema RVIO2 inicializado con config: %s", argv[1]);
        } catch (const std::exception& e) {
            Logger(ERROR, "Error inicializando RVIO2: %s", e.what());
            closeSourceManager();
            return -1;
        }
    } else if (mode == ExecutionMode::VisualOnly) {
        try {
            visualSystem = std::make_unique<vo::VisualInertialOdometry>(config);
            Logger(INFO, "Sistema visual-only inicializado");
            if (config.cam.has_imu && !config.gen.mono_inertial) {
                Logger(INFO, "gen.mono-inertial=false: se ignora la IMU para aislar la parte visual.");
            }
        } catch (const std::exception& e) {
            Logger(ERROR, "Error inicializando VisualInertialOdometry: %s", e.what());
            closeSourceManager();
            return -1;
        }
    } else if (mode == ExecutionMode::ImuOnly) {
        try {
            imuFrontend = std::make_unique<ImuFrontend>();
            imuFrontend->init(config);
            Logger(INFO, "Frontend IMU inicializado para modo inercial-only");
        } catch (const std::exception& e) {
            Logger(ERROR, "Error inicializando frontend IMU: %s", e.what());
            closeSourceManager();
            return -1;
        }
    } else {
        Logger(WARN, "No hay pipeline de estimacion activo para esta fuente; solo se registraran datos disponibles.");
    }

    bool gtEnabled = false;
#if HAVE_GT_ESTIMATOR
    GroundTruthEstimator gt;
    GroundTruthConfig gtCfg;
    GroundTruthFrame gtFrame;
    GroundTruthState gtState;
    gtEnabled = config.gen.calc_gt;
    if (gtEnabled) {
        gtCfg.enabled = true;
        if (!gt.init(gtCfg)) {
            Logger(WARN, "No se pudo inicializar GroundTruthEstimator; se desactiva GT.");
            gtEnabled = false;
        }
    }
#else
    if (config.gen.calc_gt) {
        Logger(WARN, "gt_estimator.hpp no esta disponible en esta compilacion; GT desactivado.");
    }
#endif

    if (config.gen.show && visualEnabled) {
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
    imuData lastImuRaw;
    imuData lastImuCal;
    ImuFrontendOutput lastImuState;
    bool haveLastImuRaw = false;
    bool haveImuState = false;
    double lastImuTsProcessed = -std::numeric_limits<double>::infinity();
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

        bool fatalError = false;

        for (const auto& imuSample : packet.imu_data) {
            haveLastImuRaw = true;
            lastImuRaw = imuSample;
            lastImuCal = calibrateImuSample(imuSample, &config.imu);

            if (!(imuSample.ts > lastImuTsProcessed + 1e-9)) {
                Logger(WARN, "Skipping non-increasing IMU sample ts=%.9f last=%.9f",
                       imuSample.ts, lastImuTsProcessed);
                continue;
            }

            lastImuTsProcessed = imuSample.ts;

            try {
                if (mode == ExecutionMode::Rvio2Vio && rvioSystem) {
                    rvioSystem->pushImuSample(lastImuCal);
                } else if (mode == ExecutionMode::ImuOnly && imuFrontend) {
                    lastImuState = imuFrontend->update(imuSample);
                    haveImuState = true;
                }
            } catch (const std::exception& e) {
                Logger(ERROR, "Error procesando IMU en modo %s: %s", executionModeName(mode), e.what());
                fatalError = true;
                break;
            }
        }
        if (fatalError) {
            break;
        }

#if HAVE_GT_ESTIMATOR
        if (gtEnabled && packet.new_color && packet.new_depth && !packet.color.empty() && !packet.depth.empty()) {
            try {
                gtFrame.rgb = toDisplayBgr(packet.color, config.cam.rgb_format);
                gtFrame.gray = toGray(packet.color, config.cam.rgb_format);
                gtFrame.depth_m = packet.depth;
                gtFrame.K = config.cam.K;
                gtFrame.dist = config.cam.D;
                gtFrame.timestamp_sec = static_cast<double>(packet.colorts_ms) * 1e-3;
                gtFrame.frame_number = packet.color_frame_number;
                gtState = gt.update(gtFrame);
                gt_xyz = gtState.xyz;
            } catch (const std::exception& e) {
                Logger(WARN, "Ground truth update skipped: %s", e.what());
            }
        }
#endif

        if (visualEnabled && packet.new_color && !packet.color.empty()) {
            const double frameTsSec = static_cast<double>(packet.colorts_ms) * 1e-3;
            std::optional<RvioPoseOutput> pose;
            cv::Mat debugVis;

            if (mode == ExecutionMode::Rvio2Vio && rvioSystem) {
                cv::Mat frameForRvio = toGray(packet.color, config.cam.rgb_format);

                try {
                    pose = rvioSystem->pushImageFrame(frameTsSec, packet.color_frame_number, frameForRvio);
                } catch (const std::exception& e) {
                    Logger(ERROR, "Error actualizando RVIO2: %s", e.what());
                    continue;
                }

                debugVis = toDisplayBgr(packet.color, config.cam.rgb_format);
                overlayPoseInfo(pose, "RVIO2", rvioSystem->isInitialized(), gtEnabled, gt_xyz, &debugVis);
            } else if (mode == ExecutionMode::VisualOnly && visualSystem) {
                vo::VisionFrame frame;
                frame.bgr = toDisplayBgr(packet.color, config.cam.rgb_format);
                frame.gray = toGray(packet.color, config.cam.rgb_format);
                frame.timestamp_sec = frameTsSec;
                frame.frame_number = packet.color_frame_number;

                vo::VioPoseOutput visualState;
                try {
                    visualState = visualSystem->update(
                        frame,
                        nullptr,
                        (config.gen.show || config.gen.debug) ? &debugVis : nullptr);
                } catch (const std::exception& e) {
                    Logger(ERROR, "Error actualizando visual-only: %s", e.what());
                    continue;
                }

                pose = poseFromVisualOutput(visualState, frameTsSec, packet.color_frame_number);
                if (debugVis.empty()) {
                    debugVis = frame.bgr.clone();
                }
                overlayGroundTruthInfo(gtEnabled, gt_xyz, 168, &debugVis);
            }

            if (config.gen.show && !debugVis.empty()) {
                cv::imshow("vio", debugVis);
            }

            logger.addRow(makeCsvRow(frameTsSec,
                                     pose,
                                     haveLastImuRaw ? &lastImuRaw : nullptr,
                                     haveLastImuRaw ? &lastImuCal : nullptr,
                                     gt_xyz));

            if (config.gen.debug) {
                if (pose.has_value()) {
                    Logger(DEBUG,
                           "mode=%s frame=%llu t=%.3f pose=[%.3f %.3f %.3f] rpy=[%.2f %.2f %.2f]",
                           executionModeName(mode),
                           static_cast<unsigned long long>(packet.color_frame_number),
                           frameTsSec,
                           pose->position_xyz[0], pose->position_xyz[1], pose->position_xyz[2],
                           pose->rpy_deg[0], pose->rpy_deg[1], pose->rpy_deg[2]);
                } else {
                    Logger(DEBUG,
                           "mode=%s frame=%llu t=%.3f aun sin salida valida",
                           executionModeName(mode),
                           static_cast<unsigned long long>(packet.color_frame_number),
                           frameTsSec);
                }
            }
        } else if (mode == ExecutionMode::ImuOnly && haveImuState && haveLastImuRaw) {
            const std::optional<RvioPoseOutput> pose = poseFromImuOutput(lastImuState, packet.color_frame_number);
            logger.addRow(makeCsvRow(lastImuState.timestamp_sec,
                                     pose,
                                     &lastImuRaw,
                                     &lastImuCal,
                                     gt_xyz));

            if (config.gen.debug && pose.has_value()) {
                Logger(DEBUG,
                       "mode=%s t=%.3f pose=[%.3f %.3f %.3f] rpy=[%.2f %.2f %.2f]",
                       executionModeName(mode),
                       lastImuState.timestamp_sec,
                       pose->position_xyz[0], pose->position_xyz[1], pose->position_xyz[2],
                       pose->rpy_deg[0], pose->rpy_deg[1], pose->rpy_deg[2]);
            }
        } else if (haveLastImuRaw) {
            logger.addRow(makeCsvRow(lastImuRaw.ts,
                                     std::nullopt,
                                     &lastImuRaw,
                                     &lastImuCal,
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
