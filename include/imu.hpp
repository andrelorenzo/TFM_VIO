#pragma once

#include "config.hpp"

#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void ComplementaryImuOri(imuData * imu, float alpha);
imuData calibrateImuSample(const imuData& raw, ImuCalib* calib);

// ============================================================
// Allan helpers
// ============================================================
// Se interpretan como densidades/PSD continuas. El dt se aplica en la
// discretizacion del filtro, no aqui.
static inline double qAngleFromAllan(double N_g)
{
    return N_g * N_g;
}

static inline double qBiasFromAllan(double B_g, double K_g)
{
    return B_g * B_g + K_g * K_g;
}

static inline double rAccelFromAllan(double N_a, double fs)
{
    static constexpr double G_TO_MS2 = 9.80665;
    const double sigma_sample = N_a * std::sqrt(std::max(fs, 1e-9));
    const double sigma_angle = sigma_sample / G_TO_MS2;
    return sigma_angle * sigma_angle;
}

// ============================================================
// 1D Kalman angle+bias
// ============================================================
class KalmanAngle1D
{
public:
    void begin(double q_angle_in, double q_bias_in, double r_measure_in)
    {
        q_angle = q_angle_in;
        q_bias = q_bias_in;
        r_measure = r_measure_in;

        angle = 0.0;
        bias  = 0.0;

        P00 = 1e-3;
        P01 = 0.0;
        P10 = 0.0;
        P11 = 1e-3;

        initialized = false;
    }

    void setNoise(double q_angle_in, double q_bias_in, double r_measure_in)
    {
        q_angle = q_angle_in;
        q_bias = q_bias_in;
        r_measure = r_measure_in;
    }

    void setAngle(double a)
    {
        angle = a;
        initialized = true;
    }

    double update(double gyroRate, double accelAngle, double dt)
    {
        if (!initialized) {
            setAngle(accelAngle);
        }

        const double rate = gyroRate - bias;
        angle += dt * rate;

        // Q_angle y Q_bias son intensidades continuas. La discretizacion se hace aqui.
        P00 += dt * (dt * P11 - P01 - P10 + q_angle);
        P01 -= dt * P11;
        P10 -= dt * P11;
        P11 += q_bias * dt;

        const double y = accelAngle - angle;
        const double S = P00 + r_measure;
        const double K0 = P00 / S;
        const double K1 = P10 / S;

        angle += K0 * y;
        bias  += K1 * y;

        const double P00_temp = P00;
        const double P01_temp = P01;

        P00 -= K0 * P00_temp;
        P01 -= K0 * P01_temp;
        P10 -= K1 * P00_temp;
        P11 -= K1 * P01_temp;

        return angle;
    }

    double getAngle() const { return angle; }
    double getBias() const { return bias; }

private:
    bool initialized = false;

    double angle = 0.0;
    double bias  = 0.0;

    double q_angle = 0.0;
    double q_bias = 0.0;
    double r_measure = 0.0;

    double P00 = 0.0, P01 = 0.0, P10 = 0.0, P11 = 0.0;
};

// ============================================================
// Estado completo
// ============================================================
struct imuFullState {
    double ts = 0.0;
    double dt = 0.0;
    bool stationary = false;

    cv::Vec3d rpy = {0.0, 0.0, 0.0};
    cv::Vec3d rpy_deg = {0.0, 0.0, 0.0};
    cv::Vec3d vrpy = {0.0, 0.0, 0.0};

    cv::Vec3d Axyz = {0.0, 0.0, 0.0};   // world, sin gravedad
    cv::Vec3d Vxyz = {0.0, 0.0, 0.0};   // world
    cv::Vec3d xyz  = {0.0, 0.0, 0.0};   // world

    cv::Vec3d gyro_cal = {0.0, 0.0, 0.0};
    cv::Vec3d accel_cal = {0.0, 0.0, 0.0};

    cv::Vec4d quat = {1.0, 0.0, 0.0, 0.0};
    cv::Vec3d gyro_bias_dyn = {0.0, 0.0, 0.0};

    cv::Vec3d accel_bias_body = {0.0, 0.0, 0.0};
    cv::Vec3d accel_bias_world = {0.0, 0.0, 0.0};
};

// ============================================================
// Filtro IMU: Kalman simple o Allan
// ============================================================
class ImuKalmanFullState
{
public:
    enum class Mode
    {
        Simple,
        Allan
    };

    struct SimpleParams
    {
        double q_angle = 1e-4;
        double q_bias = 1e-5;
        double r_measure = 1e-2;
    };

    struct TuningParams
    {
        double gate_g_err_ms2 = 1.0;
        double r_accel_scale = 5.0;
        double allan_q_scale = 5.0;

        double vel_damping = 0.05;

        double zupt_gyro_thresh_rad_s = 0.03;
        double zupt_accel_thresh_ms2 = 0.20;
        double gyro_bias_beta = 0.01;
        double accel_bias_beta = 0.01;
        double accel_deadband_ms2 = 0.04;

        double timing_alpha = 0.05;
        double min_dt_sec = 1e-4;
        double max_dt_sec = 0.2;
    };

    void init(ImuCalib* calib_in,
              Mode mode_in,
              const SimpleParams& simple_in = SimpleParams{},
              const TuningParams& tuning_in = TuningParams{})
    {
        if (calib_in == nullptr) {
            throw std::invalid_argument("ImuKalmanFullState::init calib_in is null");
        }
        if (calib_in->freq <= 0.0) {
            throw std::invalid_argument("ImuKalmanFullState::init calib->freq must be > 0");
        }

        calib = calib_in;
        mode = mode_in;
        simple = simple_in;
        tuning = tuning_in;

        dt_nominal = 1.0 / calib->freq;
        last_good_dt = dt_nominal;
        fs_est = calib->freq;

        configureTiltFilters();

        roll = pitch = yaw = 0.0;
        gyro_bias_z = 0.0;
        vel = cv::Vec3d(0.0, 0.0, 0.0);
        pos = cv::Vec3d(0.0, 0.0, 0.0);
        lin_acc_world = cv::Vec3d(0.0, 0.0, 0.0);
        accel_bias_body = cv::Vec3d(0.0, 0.0, 0.0);

        first_update = true;
        last_ts = 0.0;
        initialized = true;
    }

    imuFullState update(const imuData& raw)
    {
        if (!initialized || calib == nullptr) {
            throw std::runtime_error("ImuKalmanFullState::init() must be called first");
        }

        static constexpr double G = 9.80665;

        const imuData cal = calibrateImuSample(raw, calib);

        const double dt = computeDtFromTimestamp(cal.ts);
        updateEstimatedFrequency(dt);
        if (mode == Mode::Allan) {
            refreshAllanMeasurementNoise();
        }

        const double gx = cal.gyro[0];
        const double gy = cal.gyro[1];
        const double gz = cal.gyro[2];

        const double ax = cal.accel[0];
        const double ay = cal.accel[1];
        const double az = cal.accel[2];

        const double an = std::sqrt(ax * ax + ay * ay + az * az);
        const double g_err = std::fabs(an - G);

        double rollAcc = roll;
        double pitchAcc = pitch;

        if (an > 1e-9) {
            rollAcc  = std::atan2(ay, az);
            pitchAcc = std::atan2(-ax, std::sqrt(ay * ay + az * az));
        }

        if (an > 1e-9 && g_err < tuning.gate_g_err_ms2) {
            roll = kRoll.update(gx, rollAcc, dt);
            pitch = kPitch.update(gy, pitchAcc, dt);
        } else {
            roll  += (gx - kRoll.getBias()) * dt;
            pitch += (gy - kPitch.getBias()) * dt;
        }

        double wx = gx - kRoll.getBias();
        double wy = gy - kPitch.getBias();
        double wz = gz - gyro_bias_z;

        double gyro_norm = std::sqrt(wx * wx + wy * wy + wz * wz);
        bool stationary =
            (gyro_norm < tuning.zupt_gyro_thresh_rad_s) &&
            (g_err < tuning.zupt_accel_thresh_ms2);

        if (stationary) {
            const double beta = clamp01(tuning.gyro_bias_beta);
            gyro_bias_z = (1.0 - beta) * gyro_bias_z + beta * gz;
            wz = gz - gyro_bias_z;
        }

        yaw += wz * dt;

        const cv::Vec4d q_full = eulerToQuaternion(roll, pitch, yaw);
        const cv::Matx33d Rwb_tilt = eulerToRot(roll, pitch, 0.0);
        const cv::Matx33d Rbw_tilt = Rwb_tilt.t();

        const cv::Vec3d gravity_world(0.0, 0.0, G);
        const cv::Vec3d gravity_body = Rbw_tilt * gravity_world;
        const cv::Vec3d accel_residual_body = cal.accel - gravity_body;

        cv::Vec3d lin_acc_body(0.0, 0.0, 0.0);

        if (stationary) {
            const double beta = clamp01(tuning.accel_bias_beta);
            accel_bias_body =
                (1.0 - beta) * accel_bias_body +
                beta * accel_residual_body;

            vel = cv::Vec3d(0.0, 0.0, 0.0);
            lin_acc_world = cv::Vec3d(0.0, 0.0, 0.0);
        } else {
            lin_acc_body = accel_residual_body - accel_bias_body;

            for (int i = 0; i < 3; ++i) {
                if (std::fabs(lin_acc_body[i]) < tuning.accel_deadband_ms2) {
                    lin_acc_body[i] = 0.0;
                }
            }

            lin_acc_world = Rwb_tilt * lin_acc_body;

            pos += vel * dt + 0.5 * lin_acc_world * dt * dt;
            vel += lin_acc_world * dt;

            if (tuning.vel_damping > 0.0) {
                const double k = std::max(0.0, std::min(1.0, 1.0 - tuning.vel_damping * dt));
                vel *= k;
            }
        }

        imuFullState out;
        out.ts = cal.ts;
        out.dt = dt;
        out.stationary = stationary;

        out.rpy = cv::Vec3d(roll, pitch, yaw);
        out.rpy_deg = cv::Vec3d(
            roll * 180.0 / M_PI,
            pitch * 180.0 / M_PI,
            yaw * 180.0 / M_PI
        );
        out.vrpy = cv::Vec3d(wx, wy, wz);

        out.Axyz = lin_acc_world;
        out.Vxyz = vel;
        out.xyz = pos;

        out.gyro_cal = cal.gyro;
        out.accel_cal = cal.accel;
        out.quat = q_full;
        out.gyro_bias_dyn = cv::Vec3d(kRoll.getBias(), kPitch.getBias(), gyro_bias_z);
        out.accel_bias_body = accel_bias_body;
        out.accel_bias_world = Rwb_tilt * accel_bias_body;

        last_ts = cal.ts;
        first_update = false;

        return out;
    }

    void resetKinematics()
    {
        vel = cv::Vec3d(0.0, 0.0, 0.0);
        pos = cv::Vec3d(0.0, 0.0, 0.0);
        lin_acc_world = cv::Vec3d(0.0, 0.0, 0.0);
        accel_bias_body = cv::Vec3d(0.0, 0.0, 0.0);

        last_good_dt = dt_nominal;
        fs_est = (dt_nominal > 1e-9) ? (1.0 / dt_nominal) : calib->freq;
    }

private:
    static double clamp01(double x)
    {
        return std::max(0.0, std::min(1.0, x));
    }

    void configureTiltFilters()
    {
        if (mode == Mode::Allan) {
            const double q_roll = tuning.allan_q_scale * qAngleFromAllan(calib->allan_gx_nbk[0]);
            const double q_pitch = tuning.allan_q_scale * qAngleFromAllan(calib->allan_gy_nbk[0]);
            const double q_bias_roll = tuning.allan_q_scale * qBiasFromAllan(calib->allan_gx_nbk[1], calib->allan_gx_nbk[2]);
            const double q_bias_pitch = tuning.allan_q_scale * qBiasFromAllan(calib->allan_gy_nbk[1], calib->allan_gy_nbk[2]);
            const double r_roll = tuning.r_accel_scale * rAccelFromAllan(calib->allan_ay_nbk[0], fs_est);
            const double r_pitch = tuning.r_accel_scale * rAccelFromAllan(calib->allan_ax_nbk[0], fs_est);
            kRoll.begin(q_roll, q_bias_roll, r_roll);
            kPitch.begin(q_pitch, q_bias_pitch, r_pitch);
        } else {
            kRoll.begin(simple.q_angle, simple.q_bias, simple.r_measure);
            kPitch.begin(simple.q_angle, simple.q_bias, simple.r_measure);
        }
    }

    void refreshAllanMeasurementNoise()
    {
        if (mode != Mode::Allan || calib == nullptr) {
            return;
        }

        const double q_roll = tuning.allan_q_scale * qAngleFromAllan(calib->allan_gx_nbk[0]);
        const double q_pitch = tuning.allan_q_scale * qAngleFromAllan(calib->allan_gy_nbk[0]);
        const double q_bias_roll = tuning.allan_q_scale * qBiasFromAllan(calib->allan_gx_nbk[1], calib->allan_gx_nbk[2]);
        const double q_bias_pitch = tuning.allan_q_scale * qBiasFromAllan(calib->allan_gy_nbk[1], calib->allan_gy_nbk[2]);
        const double r_roll = tuning.r_accel_scale * rAccelFromAllan(calib->allan_ay_nbk[0], fs_est);
        const double r_pitch = tuning.r_accel_scale * rAccelFromAllan(calib->allan_ax_nbk[0], fs_est);
        kRoll.setNoise(q_roll, q_bias_roll, r_roll);
        kPitch.setNoise(q_pitch, q_bias_pitch, r_pitch);
    }

    double computeDtFromTimestamp(double ts)
    {
        if (first_update) {
            return dt_nominal;
        }

        const double raw_dt = ts - last_ts;
        if (std::isfinite(raw_dt) &&
            raw_dt >= tuning.min_dt_sec &&
            raw_dt <= tuning.max_dt_sec) {
            last_good_dt = raw_dt;
            return raw_dt;
        }

        return (last_good_dt > 0.0) ? last_good_dt : dt_nominal;
    }

    void updateEstimatedFrequency(double dt)
    {
        if (!std::isfinite(dt) || dt <= 0.0) {
            return;
        }

        const double inst_fs = 1.0 / dt;
        const double a = clamp01(tuning.timing_alpha);
        fs_est = (1.0 - a) * fs_est + a * inst_fs;
    }

    static cv::Vec4d eulerToQuaternion(double roll, double pitch, double yaw)
    {
        const double cr = std::cos(roll * 0.5);
        const double sr = std::sin(roll * 0.5);
        const double cp = std::cos(pitch * 0.5);
        const double sp = std::sin(pitch * 0.5);
        const double cy = std::cos(yaw * 0.5);
        const double sy = std::sin(yaw * 0.5);

        return cv::Vec4d(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );
    }

    static cv::Matx33d eulerToRot(double roll, double pitch, double yaw)
    {
        return quatToRot(eulerToQuaternion(roll, pitch, yaw));
    }

    static cv::Matx33d quatToRot(const cv::Vec4d& q)
    {
        const double w = q[0];
        const double x = q[1];
        const double y = q[2];
        const double z = q[3];

        return cv::Matx33d(
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)
        );
    }

private:
    ImuCalib* calib = nullptr;
    Mode mode = Mode::Simple;
    SimpleParams simple{};
    TuningParams tuning{};

    KalmanAngle1D kRoll;
    KalmanAngle1D kPitch;

    bool initialized = false;
    bool first_update = true;
    double last_ts = 0.0;

    double dt_nominal = 0.0;
    double last_good_dt = 0.0;
    double fs_est = 0.0;

    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    double gyro_bias_z = 0.0;

    cv::Vec3d vel = {0.0, 0.0, 0.0};
    cv::Vec3d pos = {0.0, 0.0, 0.0};
    cv::Vec3d lin_acc_world = {0.0, 0.0, 0.0};
    cv::Vec3d accel_bias_body = {0.0, 0.0, 0.0};
};
