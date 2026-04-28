#define LOGGER_IMP
#include "seconds/logger.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "CSVlogger.hpp"
#include "config.hpp"
#include "gt_estimator.hpp"
#include "imu.hpp"
#include "source_man.hpp"

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

    cv::FileNode node = fs[key];
    if (node.empty()) {
        std::cerr << "Clave YAML ausente: " << key << std::endl;
        return false;
    }

    if (node.isInt()) {
        *out = ((int)node) != 0;
        return true;
    }

    if (node.isReal()) {
        *out = std::abs((double)node) > 0.5;
        return true;
    }

    if (node.isString()) {
        std::string s = lowerCopy(trimCopy((std::string)node));

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

static bool readMat33NodeOptional(const cv::FileStorage& fs,
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
                tmp.at<double>(2, 0), tmp.at<double>(2, 1), tmp.at<double>(2, 2)
            );
            return true;
        }
    }

    return false;
}

bool parseConfig(const char * path, Config * config)
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
        Logger(DEBUG, "Successfully detected serial port input");
        config->gen.in_type = FEED_PORT;
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
    fs["imu.tocolor"] >> config->imu.T;
    config->imu.T.convertTo(config->imu.T, CV_64F);

    fs["orb.nFeatures"] >> config->orb.nFeatures;
    fs["orb.scaleFactor"] >> config->orb.scaleFactor;
    fs["orb.nLevels"] >> config->orb.nLevels;
    fs["orb.iniThFAST"] >> config->orb.iniThFAST;
    fs["orb.minThFAST"] >> config->orb.minThFAST;

    if (config->imu.freq <= 0.0) {
        std::cerr << "Frecuencia de IMU invalida" << std::endl;
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

void launchScript(const char * script)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const int ret = std::system(script);
    if (ret != 0) {
        Logger(ERROR, "Python script returned error code %d", ret);
    }
}

int main(int argc, char ** argv)
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

    ImuKalmanFullState ekf;
    try {
        const auto mode = config.gen.use_allan ? ImuKalmanFullState::Mode::Allan
                                               : ImuKalmanFullState::Mode::Simple;
        ekf.init(&config.imu, mode);
        Logger(INFO, "Filtro IMU seleccionado: %s", config.gen.use_allan ? "Allan tilt+bias" : "Simple tilt+bias");
    } catch (const std::exception& e) {
        Logger(ERROR, "Error inicializando filtro IMU: %s", e.what());
        closeSourceManager();
        return -1;
    }

    GroundTruthEstimator gt;
    GroundTruthConfig gt_cfg;
    GroundTruthFrame gt_frame;
    if (config.gen.calc_gt) {
        gt_cfg.enabled = true;
        if (!gt.init(gt_cfg)) {
            Logger(ERROR, "No se pudo inicializar GroundTruthEstimator");
            closeSourceManager();
            return -1;
        }
    }

    sourcePacket packet;
    imuData imu_raw;
    imuFullState imu_state;
    GroundTruthState gt_state;

    if (config.gen.show && config.gen.in_type != FEED_CSV && config.gen.in_type != FEED_PORT) {
        cv::namedWindow("feed", cv::WINDOW_NORMAL);
        cv::resizeWindow("feed", 1080, 720);
    }

    CsvLogger logger;
    const std::vector<std::string> csv_header = {
        "timestamp",
        "dt",
        "stationary",
        "gyro_raw_x", "gyro_raw_y", "gyro_raw_z",
        "gyro_cal_x", "gyro_cal_y", "gyro_cal_z",
        "gyro_bias_x", "gyro_bias_y", "gyro_bias_z",
        "acc_raw_x", "acc_raw_y", "acc_raw_z",
        "acc_cal_x", "acc_cal_y", "acc_cal_z",
        "acc_lin_x", "acc_lin_y", "acc_lin_z",
        "vel_x", "vel_y", "vel_z",
        "roll_deg", "pitch_deg", "yaw_deg",
        "x", "y", "z",
        "gt_x", "gt_y", "gt_z"
    };

    if (!logger.init(config.gen.output.c_str(), csv_header)) {
        Logger(ERROR, "Error creating csv file: %s", config.gen.output.c_str());
        closeSourceManager();
        cv::destroyAllWindows();
        return -1;
    }

    if (!config.gen.script.empty()) {
        std::thread(launchScript, config.gen.script.c_str()).detach();
    }

    while (true) {
        if (!getSourceManager(&packet)) {
            Logger(INFO, "No more data from source");
            break;
        }

        if (!packet.has_new_imu && !packet.has_new_color && !packet.has_new_depth) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (packet.has_new_imu) {
            imu_raw = packet.imu;
            if (config.gen.debug) {
                Logger(DEBUG,
                       "dt=%.6f stat=%d gyro=[%.4f %.4f %.4f] vel=[%.4f %.4f %.4f] pos=[%.4f %.4f %.4f]",
                       imu_state.dt,
                       imu_state.stationary ? 1 : 0,
                       imu_state.vrpy[0], imu_state.vrpy[1], imu_state.vrpy[2],
                       imu_state.Vxyz[0], imu_state.Vxyz[1], imu_state.Vxyz[2],
                       imu_state.xyz[0], imu_state.xyz[1], imu_state.xyz[2]);

          
            }
            try {
                imu_state = ekf.update(packet.imu);
            } catch (const std::exception& e) {
                Logger(ERROR, "Error actualizando filtro IMU: %s", e.what());
                break;
            }

            const std::vector<double> row = {
                imu_raw.ts,
                imu_state.dt,
                imu_state.stationary ? 1.0 : 0.0,
                imu_raw.gyro[0], imu_raw.gyro[1], imu_raw.gyro[2],
                imu_state.gyro_cal[0], imu_state.gyro_cal[1], imu_state.gyro_cal[2],
                imu_state.gyro_bias_dyn[0], imu_state.gyro_bias_dyn[1], imu_state.gyro_bias_dyn[2],
                imu_raw.accel[0], imu_raw.accel[1], imu_raw.accel[2],
                imu_state.accel_cal[0], imu_state.accel_cal[1], imu_state.accel_cal[2],
                imu_state.Axyz[0], imu_state.Axyz[1], imu_state.Axyz[2],
                imu_state.Vxyz[0], imu_state.Vxyz[1], imu_state.Vxyz[2],
                imu_state.rpy_deg[0], imu_state.rpy_deg[1], imu_state.rpy_deg[2],
                imu_state.xyz[0], imu_state.xyz[1], imu_state.xyz[2],
                gt_state.xyz[0], gt_state.xyz[1], gt_state.xyz[2]
            };
            logger.addRow(row);


        }

        if (config.gen.calc_gt && packet.has_depth && packet.has_new_color && packet.has_new_depth) {
            gt_frame.rgb = packet.color_bgr;
            gt_frame.gray = packet.gray;
            gt_frame.depth_m = packet.depth_m;
            gt_frame.K = packet.K;
            gt_frame.dist = packet.dist;
            gt_frame.timestamp_sec = packet.timestamp_sec;
            gt_frame.frame_number = packet.frame_number;
            gt_state = gt.update(gt_frame);
        }

        if (config.gen.show && !packet.color_bgr.empty() && packet.has_new_color) {
            cv::Mat vis = packet.color_bgr.clone();
            std::ostringstream oss0, oss1, oss2, oss3, oss4, oss5, oss6;
            oss0 << "Mode  : " << (config.gen.use_allan ? "Allan" : "Simple");
            oss1 << std::fixed << std::setprecision(2) << "Roll  : " << imu_state.rpy_deg[0] << " deg";
            oss2 << std::fixed << std::setprecision(2) << "Pitch : " << imu_state.rpy_deg[1] << " deg";
            oss3 << std::fixed << std::setprecision(2) << "Yaw   : " << imu_state.rpy_deg[2] << " deg";
            oss4 << std::fixed << std::setprecision(3) << "GT xyz : [" << gt_state.xyz[0] << ", " << gt_state.xyz[1] << ", " << gt_state.xyz[2] << "]";
            oss5 << std::fixed << std::setprecision(3) << "IMU xyz: [" << imu_state.xyz[0] << ", " << imu_state.xyz[1] << ", " << imu_state.xyz[2] << "]";
            oss6 << std::fixed << std::setprecision(3) << "Vel    : [" << imu_state.Vxyz[0] << ", " << imu_state.Vxyz[1] << ", " << imu_state.Vxyz[2] << "]";

            cv::putText(vis, oss0.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(vis, oss1.str(), cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(vis, oss2.str(), cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            cv::putText(vis, oss3.str(), cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            if (config.cam.has_imu) {
                cv::putText(vis, oss5.str(), cv::Point(20, 150), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                cv::putText(vis, oss6.str(), cv::Point(20, 180), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
            }
            if (config.gen.calc_gt) {
                cv::putText(vis, oss4.str(), cv::Point(20, 210), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            }

            cv::imshow("feed", vis);
        }

        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
        printf("\033[2J\033[1;1H");

    }

    logger.close();
    closeSourceManager();
    cv::destroyAllWindows();
    return 0;
}
