#ifndef CONFIG_H_
#define CONFIG_H_

#include <iostream>
#include <iomanip>
#include <string>
#include <opencv2/opencv.hpp>
#include "seconds/logger.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


typedef enum{
    FEED_BAG = 0,
    FEED_RSCAM,
    FEED_RTSP,
    FEED_PCCAM,
    FEED_CSV,
    FEED_PORT,
    FEED_MP4
} InputType;

struct GeneralParams {
    bool show = false;
    bool mono_inertial = false;
    InputType in_type = FEED_BAG;
    std::string input = "";
    std::string output = "";
    std::string script = "";
    bool calc_gt = false;
    bool debug = false;
    bool use_allan = false;
};

struct CameraParams {
    bool has_imu = false;
    int w = 0;
    int h = 0;
    int fps = 0;
    int rgb_format = 0;
    cv::Mat K;
    cv::Mat D;
};

struct ImuCalib {
    cv::Vec3d gv = {0.0, 0.0, 9.80665};
    cv::Vec3d gyro_bias = {0.0, 0.0, 0.0};
    cv::Vec3d accel_bias = {0.0, 0.0, 0.0};

    cv::Vec3d allan_gx_nbk = {0.0, 0.0, 0.0};
    cv::Vec3d allan_gy_nbk = {0.0, 0.0, 0.0};
    cv::Vec3d allan_gz_nbk = {0.0, 0.0, 0.0};

    cv::Vec3d allan_ax_nbk = {0.0, 0.0, 0.0};
    cv::Vec3d allan_ay_nbk = {0.0, 0.0, 0.0};
    cv::Vec3d allan_az_nbk = {0.0, 0.0, 0.0};

    cv::Matx33d gyro_scale_misalignment = cv::Matx33d::eye();
    cv::Matx33d accel_scale_misalignment = cv::Matx33d::eye();

    double freq = 0.0;
    cv::Mat T;

    cv::Vec3d gyroAllanN() const {
        return cv::Vec3d(allan_gx_nbk[0], allan_gy_nbk[0], allan_gz_nbk[0]);
    }

    cv::Vec3d gyroAllanK() const {
        return cv::Vec3d(allan_gx_nbk[2], allan_gy_nbk[2], allan_gz_nbk[2]);
    }

    cv::Vec3d accelAllanN() const {
        return cv::Vec3d(allan_ax_nbk[0], allan_ay_nbk[0], allan_az_nbk[0]);
    }

    cv::Vec3d accelAllanK() const {
        return cv::Vec3d(allan_ax_nbk[2], allan_ay_nbk[2], allan_az_nbk[2]);
    }

    bool hasValidAllan() const {
        const cv::Vec3d Ng = gyroAllanN();
        const cv::Vec3d Na = accelAllanN();
        const cv::Vec3d Kg = gyroAllanK();
        const cv::Vec3d Ka = accelAllanK();
        return Ng[0] > 0.0 && Ng[1] > 0.0 && Ng[2] > 0.0 &&
               Na[0] > 0.0 && Na[1] > 0.0 && Na[2] > 0.0 &&
               Kg[0] > 0.0 && Kg[1] > 0.0 && Kg[2] > 0.0 &&
               Ka[0] > 0.0 && Ka[1] > 0.0 && Ka[2] > 0.0;
    }

    bool hasValidGravity() const {
        return cv::norm(gv) > 1e-6;
    }
};

struct imuData {
    double ts = 0.0;
    cv::Vec3d gyro = {0.0, 0.0, 0.0};   // rad/s
    cv::Vec3d accel = {0.0, 0.0, 0.0};  // m/s^2
    cv::Vec3d orir = {0.0, 0.0, 0.0};   // rad (roll, pitch, yaw)
    cv::Vec3d orid = {0.0, 0.0, 0.0};   // deg (roll, pitch, yaw)
    cv::Vec4d quat = {1.0, 0.0, 0.0, 0.0}; // qw, qx, qy, qz
    cv::Vec3d gyro_bias_dyn = {0.0, 0.0, 0.0};
};

struct OrbParams {
    int nFeatures = 0;
    float scaleFactor = 1.2f;
    int nLevels = 0;
    int iniThFAST = 0;
    int minThFAST = 0;
};

struct Config {
    GeneralParams gen;
    CameraParams cam;
    ImuCalib imu;
    OrbParams orb;

    void print(std::ostream& os = std::cout) const {
        os << std::boolalpha;
        os << std::fixed << std::setprecision(6);

        os << "\n================ CONFIG ================\n";

        os << "\n[General]\n";
        os << "  show           : " << gen.show << "\n";
        os << "  mono_inertial  : " << gen.mono_inertial << "\n";
        os << "  input type     : " << inputTypeToString(gen.in_type) << "\n";
        os << "  input          : " << gen.input << "\n";
        os << "  output         : " << gen.output << "\n";
        os << "  calc_gt        : " << gen.calc_gt << "\n";
        os << "  debug          : " << gen.debug << "\n";
        os << "  use_allan      : " << gen.use_allan << "\n";

        os << "\n[Camera]\n";
        os << "  has_imu        : " << cam.has_imu << "\n";
        os << "  width          : " << cam.w << "\n";
        os << "  height         : " << cam.h << "\n";
        os << "  fps            : " << cam.fps << "\n";
        os << "  rgb_format     : " << cam.rgb_format;
        if (cam.rgb_format == 0) os << " (BGR)\n";
        else if (cam.rgb_format == 1) os << " (RGB)\n";
        else os << " (unknown)\n";

        os << "  K              :\n";
        printMat(os, cam.K, 6);

        os << "  D              :\n";
        printMat(os, cam.D, 6);

        os << "\n[IMU]\n";
        os << "  gravity_world  : [" << imu.gv[0] << ", " << imu.gv[1] << ", " << imu.gv[2] << "]\n";
        os << "  gyro_bias      : [" << imu.gyro_bias[0] << ", " << imu.gyro_bias[1] << ", " << imu.gyro_bias[2] << "]\n";
        os << "  accel_bias     : [" << imu.accel_bias[0] << ", " << imu.accel_bias[1] << ", " << imu.accel_bias[2] << "]\n";
        os << "  allan_gx(NBK)  : [" << imu.allan_gx_nbk[0] << ", " << imu.allan_gx_nbk[1] << ", " << imu.allan_gx_nbk[2] << "]\n";
        os << "  allan_gy(NBK)  : [" << imu.allan_gy_nbk[0] << ", " << imu.allan_gy_nbk[1] << ", " << imu.allan_gy_nbk[2] << "]\n";
        os << "  allan_gz(NBK)  : [" << imu.allan_gz_nbk[0] << ", " << imu.allan_gz_nbk[1] << ", " << imu.allan_gz_nbk[2] << "]\n";
        os << "  allan_ax(NBK)  : [" << imu.allan_ax_nbk[0] << ", " << imu.allan_ax_nbk[1] << ", " << imu.allan_ax_nbk[2] << "]\n";
        os << "  allan_ay(NBK)  : [" << imu.allan_ay_nbk[0] << ", " << imu.allan_ay_nbk[1] << ", " << imu.allan_ay_nbk[2] << "]\n";
        os << "  allan_az(NBK)  : [" << imu.allan_az_nbk[0] << ", " << imu.allan_az_nbk[1] << ", " << imu.allan_az_nbk[2] << "]\n";
        os << "  freq           : " << imu.freq << "\n";
        os << "  gyro_sm        :\n";
        printMat(os, cv::Mat(imu.gyro_scale_misalignment), 6);
        os << "  accel_sm       :\n";
        printMat(os, cv::Mat(imu.accel_scale_misalignment), 6);
        os << "  T              :\n";
        printMat(os, imu.T, 6);

        os << "\n[ORB]\n";
        os << "  nFeatures      : " << orb.nFeatures << "\n";
        os << "  scaleFactor    : " << orb.scaleFactor << "\n";
        os << "  nLevels        : " << orb.nLevels << "\n";
        os << "  iniThFAST      : " << orb.iniThFAST << "\n";
        os << "  minThFAST      : " << orb.minThFAST << "\n";

        os << "\n========================================\n";
    }

private:
    static const char* inputTypeToString(InputType type) {
        switch (type) {
            case FEED_BAG:   return "FEED_BAG";
            case FEED_RSCAM: return "FEED_RSCAM";
            case FEED_RTSP:  return "FEED_RTSP";
            case FEED_PCCAM: return "FEED_PCCAM";
            case FEED_CSV:   return "FEED_CSV";
            case FEED_PORT:  return "FEED_PORT";
            case FEED_MP4:   return "FEED_MP4";
            default:         return "UNKNOWN";
        }
    }

    static void printMat(std::ostream& os, const cv::Mat& mat, int precision = 6) {
        if (mat.empty()) {
            os << "    <empty>\n";
            return;
        }

        cv::Mat tmp;
        mat.convertTo(tmp, CV_64F);

        os << std::fixed << std::setprecision(precision);
        for (int r = 0; r < tmp.rows; ++r) {
            os << "    [ ";
            for (int c = 0; c < tmp.cols; ++c) {
                os << std::setw(12) << tmp.at<double>(r, c);
                if (c < tmp.cols - 1) os << " ";
            }
            os << " ]\n";
        }
    }
};

#endif // CONFIG_H_
