#include "imu.hpp"

#include <cmath>



imuData calibrateImuSample(const imuData& raw, ImuCalib* calib)
{
    if (calib == nullptr) {
        throw std::invalid_argument("calibrateImuSample: calib is null");
    }

    imuData out{};
    out.ts = raw.ts;

    const cv::Vec3d gyro_unbiased  = raw.gyro  - calib->gyro_bias;
    const cv::Vec3d accel_unbiased = raw.accel - calib->accel_bias;

    out.gyro  = calib->gyro_scale_misalignment  * gyro_unbiased;
    out.accel = calib->accel_scale_misalignment * accel_unbiased;
    return out;
}

static inline double rad2deg(double x)
{
    return x * 180.0 / M_PI;
}

static inline double wrapPi(double a)
{
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}

void ComplementaryImuOri(imuData * imu, float alpha)
{
    if (imu == nullptr) return;

    static bool initialized = false;
    static double last_ts = 0.0;
    static cv::Vec3d ori = {0.0, 0.0, 0.0};

    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;

    const double ax = imu->accel[0];
    const double ay = imu->accel[1];
    const double az = imu->accel[2];

    const double gx = imu->gyro[0];
    const double gy = imu->gyro[1];
    const double gz = imu->gyro[2];

    // Gravedad en +Z cuando el sensor está horizontal:
    // accel ~ [0, 0, +g]
    // roll  alrededor de X
    // pitch alrededor de Y
    // yaw   alrededor de Z no es observable con accel
    const double roll_acc = std::atan2(ay, az);
    const double pitch_acc = std::atan2(-ax, std::sqrt(ay * ay + az * az));

    if (!initialized) {
        ori[0] = roll_acc;
        ori[1] = pitch_acc;
        ori[2] = 0.0;
        last_ts = imu->ts;
        initialized = true;
    } else {
        double dt = imu->ts - last_ts;
        if (dt <= 0.0 || dt > 1.0) {
            dt = 0.0;
        }
        last_ts = imu->ts;

        const double roll_gyro = ori[0] + gx * dt;
        const double pitch_gyro = ori[1] + gy * dt;
        const double yaw_gyro = ori[2] + gz * dt;

        ori[0] = alpha * roll_gyro + (1.0 - alpha) * roll_acc;
        ori[1] = alpha * pitch_gyro + (1.0 - alpha) * pitch_acc;
        ori[2] = yaw_gyro;
    }

    ori[0] = wrapPi(ori[0]);
    ori[1] = wrapPi(ori[1]);
    ori[2] = wrapPi(ori[2]);

    imu->orir = ori;
    imu->orid = cv::Vec3d(rad2deg(ori[0]), rad2deg(ori[1]), rad2deg(ori[2]));
}
