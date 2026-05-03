#include "librealsense2/rs.hpp"
#include "realsense.h"
#include "mahony.h"

#define LOGGER_IMP
#include "seconds/logger.h"

#include <opencv2/opencv.hpp>

#include <mutex>
#include <thread>
#include <chrono>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstdio>


struct RSData {
    double ts_ms = -1.0;

    float gyro[3] = {0.0f, 0.0f, 0.0f};   // rad/s
    float acc[3]  = {0.0f, 0.0f, 0.0f};   // m/s^2

    float roll = 0.0f;    // rad   (rotación sobre Z)
    float pitch = 0.0f;   // rad   (rotación sobre X)
    float yaw = 0.0f;     // rad   (rotación sobre Y)

    cv::Mat color;
    bool has_color = false;
};

struct AttitudeEstimatorD435i
{
    // Convención usada:
    // +X derecha
    // +Y abajo
    // +Z delante
    //
    // Ángulos:
    // roll  -> giro sobre Z
    // pitch -> giro sobre X
    // yaw   -> giro sobre Y

    double roll  = 0.0;
    double pitch = 0.0;
    double yaw   = 0.0;

    double alpha = 0.98;
    bool initialized = false;
    double last_ts = 0.0; // segundos

    void update(const double acc[3], const double gyro[3], double timestamp_s)
    {
        const double ax_raw = acc[0];
        const double ay_raw = acc[1];
        const double az_raw = acc[2];

        const double gx = gyro[0]; // rotación sobre X -> pitch
        const double gy = gyro[1]; // rotación sobre Y -> yaw
        const double gz = gyro[2]; // rotación sobre Z -> roll

        double dt = 0.0;
        if (initialized)
        {
            dt = timestamp_s - last_ts;
            if (dt < 0.0) dt = 0.0;
            if (dt > 0.1) dt = 0.1; // protección simple ante saltos
        }
        last_ts = timestamp_s;

        const double norm = std::sqrt(ax_raw * ax_raw + ay_raw * ay_raw + az_raw * az_raw);

        if (norm < 1e-9)
        {
            if (initialized && dt > 0.0)
            {
                pitch += gx * dt;
                yaw   += gy * dt;
                roll  += gz * dt;
            }
            else
            {
                initialized = true;
            }
            return;
        }

        const double ax = ax_raw / norm;
        const double ay = ay_raw / norm;
        const double az = az_raw / norm;

        // Estimación por gravedad coherente con la D435i
        const double roll_acc  = std::atan2(-ax, std::sqrt(ay * ay + az * az));
        const double pitch_acc = std::atan2( az, -ay );

        if (!initialized)
        {
            roll  = roll_acc;
            pitch = pitch_acc;
            yaw   = 0.0;
            initialized = true;
            return;
        }

        const double pitch_gyro = pitch + gx * dt;
        const double roll_gyro  = roll  + gz * dt;
        const double yaw_gyro   = yaw   + gy * dt;

        pitch = alpha * pitch_gyro + (1.0 - alpha) * pitch_acc;
        roll  = alpha * roll_gyro  + (1.0 - alpha) * roll_acc;
        yaw   = yaw_gyro; // sin referencia absoluta
    }
};

static std::mutex mut;
static RSData rs_data;
static AttitudeEstimatorD435i imu;
static Mahony mahony;

void getRealsenseData();

cv::Mat eulerToRotationMatrix(float roll, float pitch, float yaw) {
    float cr = std::cos(roll),  sr = std::sin(roll);
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cy = std::cos(yaw),   sy = std::sin(yaw);

    cv::Mat Rx = (cv::Mat_<double>(3,3) <<
        1, 0, 0,
        0, cr, -sr,
        0, sr, cr);

    cv::Mat Ry = (cv::Mat_<double>(3,3) <<
        cp, 0, sp,
        0, 1, 0,
        -sp, 0, cp);

    cv::Mat Rz = (cv::Mat_<double>(3,3) <<
        cy, -sy, 0,
        sy,  cy, 0,
        0,   0,  1);

    return Rz * Ry * Rx;
}
void drawOrientationOpenCV(cv::Mat& img, float roll_rad, float pitch_rad, float yaw_rad) {
    if (img.empty()) {
        return;
    }

    cv::Mat R = eulerToRotationMatrix(roll_rad, pitch_rad, yaw_rad);
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);

    cv::Mat K = rs_cam::GetK();
    cv::Mat distCoeffs = rs_cam::GetDist();
    cv::Mat tvec = (cv::Mat_<double>(3,1) << 0.0, 0.0, 4.0);

    std::vector<cv::Point3f> axes = {
        cv::Point3f(0.0f, 0.0f, 0.0f),
        cv::Point3f(0.8f, 0.0f, 0.0f),
        cv::Point3f(0.0f, 0.8f, 0.0f),
        cv::Point3f(0.0f, 0.0f, 0.8f)
    };

    std::vector<cv::Point2f> pts;
    cv::projectPoints(axes, rvec, tvec, K, distCoeffs, pts);

    cv::Point2f origin = pts[0];

    cv::line(img, origin, pts[1], cv::Scalar(0,   0, 255), 3);
    cv::line(img, origin, pts[2], cv::Scalar(0, 255,   0), 3);
    cv::line(img, origin, pts[3], cv::Scalar(255, 0,   0), 3);

    cv::circle(img, origin, 4, cv::Scalar(255, 255, 255), -1);

    cv::putText(img, "X", pts[1], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2);
    cv::putText(img, "Y", pts[2], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
    cv::putText(img, "Z", pts[3], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0), 2);

    char text[256];
    std::snprintf(text, sizeof(text), "R: %.2f  P: %.2f  Y: %.2f", roll_rad * 57.2958, pitch_rad * 57.2958, yaw_rad * 57.2958);
    cv::putText(img, text, cv::Point(20, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
}

void exit_callback() {
    using namespace std::chrono_literals;
    while (true) {
        int ret = getc(stdin);
        if (ret != '\n') {
            exit(1);
        }
        std::this_thread::sleep_for(10ms);
    }
}

int main() {
    LoggerSetVerbsity(DEBUG);
    Logger(INFO, "All right!!");

    std::thread realsense(getRealsenseData);
    std::thread exitcb(exit_callback);
    realsense.detach();
    exitcb.detach();

    while (true) {
        system("cls");
        cv::Mat frame;
        float gx, gy, gz;
        float ax, ay, az;
        float roll, pitch, yaw;
        double ts_ms;
        bool hasColor = false;

        {
            std::lock_guard<std::mutex> lock(mut);
            frame = rs_data.color.clone();
            hasColor = rs_data.has_color;

            gx = rs_data.gyro[0];
            gy = rs_data.gyro[1];
            gz = rs_data.gyro[2];

            ax = rs_data.acc[0];
            ay = rs_data.acc[1];
            az = rs_data.acc[2];

            roll = rs_data.roll;
            pitch = rs_data.pitch;
            yaw = rs_data.yaw;
            ts_ms = rs_data.ts_ms;
        }

        Logger(INFO, "TS(ms): %f (%f Hz)", ts_ms, 1000.0f/ts_ms);
        Logger(INFO, "GYRO(rad/s): %f %f %f", gx, gy, gz);
        Logger(INFO, "ACC(m/s2): %f %f %f", ax, ay, az);
        Logger(INFO, "ROLL: %f, PITCH: %f, YAW: %f", roll, pitch, yaw);

        if (hasColor && !frame.empty()) {
            drawOrientationOpenCV(frame, roll, pitch, yaw);
            cv::imshow("RealSense Color", frame);
        }

        int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    cv::destroyAllWindows();
    return 0;
}

void getRealsenseData() {
    const char* bagfile = "C:/Users/ajlorenzo/Documents/AAS_projects/TFM_Vision_v2/Jetson_files/20260122_182421.bag";

    rs2::pipeline pipe;
    bool bag_started = false;

    bool has_gyro = false;
    bool has_acc = false;

    float gx = 0.0f, gy = 0.0f, gz = 0.0f;   // camera frame
    float ax = 0.0f, ay = 0.0f, az = 0.0f;   // camera frame

    double last_gyro_ts_ms = -1.0;
    double last_acc_ts_ms = -1.0;

    while (true) {
        system("cls");
        if (!bag_started) {
            rs2::config cfg;
            cfg.enable_device_from_file(bagfile);

            rs2::pipeline_profile profile = pipe.start(cfg);
            rs2::device bag_dev = profile.get_device();

            if (bag_dev.is<rs2::playback>()) {
                auto pb = bag_dev.as<rs2::playback>();
                pb.set_real_time(true);
            } else {
                Logger(ERROR, "device no es playback");
            }

            Logger(INFO, "Bag file connected: %s", bagfile);
            bag_started = true;
        }

        rs2::frameset frames;
        if (!pipe.try_wait_for_frames(&frames, 5000)) {
            Logger(INFO, "Fin del .bag o timeout.");
            break;
        }

        for (auto frame : frames) {
            auto stream_prof = frame.get_profile().as<rs2::stream_profile>();
            rs2_stream stream_type = stream_prof.stream_type();
            double timestamp_ms = frame.get_timestamp();

            if (stream_type == RS2_STREAM_GYRO) {
                rs2::motion_frame mf = frame.as<rs2::motion_frame>();
                auto md = mf.get_motion_data();

                gx = md.x;
                gy = md.y;
                gz = md.z;
                has_gyro = true;

                if (last_gyro_ts_ms > 0.0) {
                    double dt_gyro_ms = timestamp_ms - last_gyro_ts_ms;
                    double dt_gyro_s = dt_gyro_ms * 1e-3;
                    double gyro_hz = (dt_gyro_s > 1e-9) ? (1.0 / dt_gyro_s) : 0.0;

                    Logger(INFO,
                           "GYRO dt: %.3f ms | %.3f Hz | ts: %.3f ms | data: %.6f %.6f %.6f",
                           dt_gyro_ms, gyro_hz, timestamp_ms, gx, gy, gz);

                    if (has_acc && dt_gyro_s > 1e-4 && dt_gyro_s < 0.1) {
                        mahony.setSampleFrequency(static_cast<float>(gyro_hz));

                        const float gx_m = gz;    // rotación sobre Zc -> Xm
                        const float gy_m = gx;    // rotación sobre Xc -> Ym
                        const float gz_m = -gy;   // rotación sobre -Yc -> Zm

                        const float ax_m = az;    // Zc -> Xm
                        const float ay_m = ax;    // Xc -> Ym
                        const float az_m = -ay;   // -Yc -> Zm

                        mahony.updateIMU(gx_m, gy_m, gz_m, ax_m, ay_m, az_m);

                        const float roll_cam  = mahony.getRollRadians();
                        const float pitch_cam = mahony.getPitchRadians();
                        const float yaw_cam   = 0.0f;

                        {
                            std::lock_guard<std::mutex> lock(mut);
                            rs_data.ts_ms = dt_gyro_ms;

                            rs_data.gyro[0] = gx;
                            rs_data.gyro[1] = gy;
                            rs_data.gyro[2] = gz;

                            rs_data.acc[0] = ax;
                            rs_data.acc[1] = ay;
                            rs_data.acc[2] = az;

                            rs_data.roll  = roll_cam;
                            rs_data.pitch = pitch_cam;
                            rs_data.yaw   = yaw_cam;
                        }
                    }
                }

                last_gyro_ts_ms = timestamp_ms;
            }
            else if (stream_type == RS2_STREAM_ACCEL) {
                rs2::motion_frame mf = frame.as<rs2::motion_frame>();
                auto md = mf.get_motion_data();

                ax = md.x;
                ay = md.y;
                az = md.z;
                has_acc = true;

                if (last_acc_ts_ms > 0.0) {
                    double dt_acc_ms = timestamp_ms - last_acc_ts_ms;
                    double dt_acc_s = dt_acc_ms * 1e-3;
                    double acc_hz = (dt_acc_s > 1e-9) ? (1.0 / dt_acc_s) : 0.0;

                    Logger(INFO,
                           "ACC  dt: %.3f ms | %.3f Hz | ts: %.3f ms | data: %.6f %.6f %.6f",
                           dt_acc_ms, acc_hz, timestamp_ms, ax, ay, az);
                }

                last_acc_ts_ms = timestamp_ms;

                {
                    std::lock_guard<std::mutex> lock(mut);
                    rs_data.acc[0] = ax;
                    rs_data.acc[1] = ay;
                    rs_data.acc[2] = az;
                }
            }
            else if (stream_type == RS2_STREAM_COLOR) {
                auto vf = frame.as<rs2::video_frame>();

                int w = vf.get_width();
                int h = vf.get_height();

                cv::Mat rgb(h, w, CV_8UC3, (void*)vf.get_data(), cv::Mat::AUTO_STEP);
                cv::Mat bgr;
                cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

                {
                    std::lock_guard<std::mutex> lock(mut);
                    rs_data.color = bgr;
                    rs_data.has_color = true;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    pipe.stop();
}