#include <mutex>
#include <optional>
#include <string>

#include <opencv2/core.hpp>

#include "rvio2/System.h"

struct AppPoseEstimate
{
    double timestamp = 0.0;
    float px = 0.0f;
    float py = 0.0f;
    float pz = 0.0f;
    float qx = 0.0f;
    float qy = 0.0f;
    float qz = 0.0f;
    float qw = 1.0f;
};


class Rvio2Adapter
{
public:
    explicit Rvio2Adapter(const std::string& configPath)
        : system_(configPath)
    {
    }

    void PushImuSample(double timestampSec,
                       float gx, float gy, float gz,
                       float ax, float ay, float az)
    {
        std::scoped_lock lock(mutex_);

        RVIO2::ImuData* data = new RVIO2::ImuData();
        data->Timestamp = timestampSec;
        data->AngularVel << gx, gy, gz;
        data->LinearAccel << ax, ay, az;
        data->TimeInterval = lastImuTimestampSec_ < 0.0 ? 0.0 : timestampSec-lastImuTimestampSec_;

        lastImuTimestampSec_ = timestampSec;
        system_.PushImuData(data);
    }

    std::optional<AppPoseEstimate> PushImageFrame(double timestampSec, const cv::Mat& image)
    {
        std::scoped_lock lock(mutex_);

        RVIO2::ImageData* data = new RVIO2::ImageData();
        data->Timestamp = timestampSec;
        data->Image = image.clone();
        system_.PushImageData(data);

        RVIO2::PoseEstimate pose;
        if (!system_.run(&pose))
            return std::nullopt;

        return ConvertPose(pose);
    }

private:
    static AppPoseEstimate ConvertPose(const RVIO2::PoseEstimate& pose)
    {
        AppPoseEstimate out;
        out.timestamp = pose.Timestamp;
        out.px = pose.Position(0);
        out.py = pose.Position(1);
        out.pz = pose.Position(2);
        out.qx = pose.Quaternion(0);
        out.qy = pose.Quaternion(1);
        out.qz = pose.Quaternion(2);
        out.qw = pose.Quaternion(3);
        return out;
    }

private:
    std::mutex mutex_;
    double lastImuTimestampSec_ = -1.0;
    RVIO2::System system_;
};


/*
Uso tipico dentro de tu proyecto:

Rvio2Adapter vio("config/rvio2_euroc.yaml");

// Callback IMU
vio.PushImuSample(imuTimestampSec, gx, gy, gz, ax, ay, az);

// Callback camara
if (auto pose = vio.PushImageFrame(imageTimestampSec, grayOrBgrImage))
{
    // Ya hay una estimacion valida.
    // pose->px, pose->py, pose->pz
    // pose->qx, pose->qy, pose->qz, pose->qw
}

Notas:
- `PushImageFrame()` puede devolver `std::nullopt` al principio si aun no hay IMU suficiente.
- Tambien devolvera `std::nullopt` durante la fase de inicializacion estacionaria.
- Si usas callbacks en hilos distintos, este adapter ya protege el acceso con mutex.
- El timestamp debe ir en segundos.
*/
