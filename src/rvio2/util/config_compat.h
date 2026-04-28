#ifndef RVIO2_CONFIG_COMPAT_H
#define RVIO2_CONFIG_COMPAT_H

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>

#include <Eigen/Core>

#include <opencv2/core.hpp>

namespace RVIO2 {
namespace config_compat {

inline std::string LowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline cv::FileNode FindNode(const cv::FileStorage& fs, std::initializer_list<const char*> keys)
{
    for (const char* key : keys) {
        const cv::FileNode node = fs[key];
        if (!node.empty()) {
            return node;
        }
    }
    return cv::FileNode();
}

inline bool ReadBool(const cv::FileStorage& fs,
                     std::initializer_list<const char*> keys,
                     bool fallback = false)
{
    const cv::FileNode node = FindNode(fs, keys);
    if (node.empty()) {
        return fallback;
    }
    if (node.isInt()) {
        return static_cast<int>(node) != 0;
    }
    if (node.isReal()) {
        return std::abs(static_cast<double>(node)) > 0.5;
    }
    if (node.isString()) {
        const std::string value = LowerCopy(static_cast<std::string>(node));
        return (value == "true" || value == "1" || value == "yes" || value == "on");
    }
    return fallback;
}

inline double ReadDouble(const cv::FileStorage& fs,
                         std::initializer_list<const char*> keys,
                         double fallback = 0.0)
{
    const cv::FileNode node = FindNode(fs, keys);
    if (node.empty()) {
        return fallback;
    }
    if (node.isInt() || node.isReal()) {
        return static_cast<double>(node);
    }
    if (node.isString()) {
        try {
            return std::stod(static_cast<std::string>(node));
        } catch (...) {
            return fallback;
        }
    }
    return fallback;
}

inline cv::Mat ReadMat(const cv::FileStorage& fs, std::initializer_list<const char*> keys);

inline float ReadFloat(const cv::FileStorage& fs,
                       std::initializer_list<const char*> keys,
                       float fallback = 0.0f)
{
    const double missing = std::numeric_limits<double>::quiet_NaN();
    const double value = ReadDouble(fs, keys, missing);
    if (std::isfinite(value)) {
        return static_cast<float>(value);
    }

    cv::Mat K = ReadMat(fs, {"cam.K"});
    if (!K.empty()) {
        K = K.reshape(1, 3);
        K.convertTo(K, CV_64F);
        for (const char* key : keys) {
            const std::string skey(key);
            if (skey == "cam.fx" || skey == "Camera.fx") {
                return static_cast<float>(K.at<double>(0, 0));
            }
            if (skey == "cam.fy" || skey == "Camera.fy") {
                return static_cast<float>(K.at<double>(1, 1));
            }
            if (skey == "cam.cx" || skey == "Camera.cx") {
                return static_cast<float>(K.at<double>(0, 2));
            }
            if (skey == "cam.cy" || skey == "Camera.cy") {
                return static_cast<float>(K.at<double>(1, 2));
            }
        }
    }

    return fallback;
}

inline int ReadInt(const cv::FileStorage& fs,
                   std::initializer_list<const char*> keys,
                   int fallback = 0)
{
    return static_cast<int>(std::lround(ReadDouble(fs, keys, fallback)));
}

inline cv::Mat ReadMat(const cv::FileStorage& fs, std::initializer_list<const char*> keys)
{
    const cv::FileNode node = FindNode(fs, keys);
    if (node.empty()) {
        return cv::Mat();
    }

    cv::Mat out;
    node >> out;
    return out;
}

inline cv::Vec3d ReadVec3(const cv::FileStorage& fs,
                          std::initializer_list<const char*> keys,
                          const cv::Vec3d& fallback = cv::Vec3d(0.0, 0.0, 0.0))
{
    cv::Mat value = ReadMat(fs, keys);
    if (value.empty()) {
        return fallback;
    }

    value = value.reshape(1, 1);
    value.convertTo(value, CV_64F);
    if (value.total() != 3) {
        return fallback;
    }

    return cv::Vec3d(value.at<double>(0, 0),
                     value.at<double>(0, 1),
                     value.at<double>(0, 2));
}

inline double AveragePositive(std::initializer_list<double> values, double fallback)
{
    double sum = 0.0;
    int count = 0;
    for (double value : values) {
        if (std::isfinite(value) && value > 0.0) {
            sum += value;
            ++count;
        }
    }
    return (count > 0) ? (sum / static_cast<double>(count)) : fallback;
}

inline double ReadAllanEntry(const cv::FileStorage& fs, const char* key, int index, double fallback)
{
    const cv::Vec3d value = ReadVec3(fs, {key}, cv::Vec3d(fallback, fallback, fallback));
    if (index < 0 || index > 2) {
        return fallback;
    }
    const double out = value[index];
    return (std::isfinite(out) && out > 0.0) ? out : fallback;
}

inline double ReadGravityMagnitude(const cv::FileStorage& fs, double fallback = 9.80665)
{
    const double scalar = ReadDouble(fs, {"imu.nG", "IMU.nG", "imu.gravity"}, -1.0);
    if (scalar > 0.0) {
        return scalar;
    }

    const cv::Vec3d gravity = ReadVec3(fs, {"imu.gv"}, cv::Vec3d(0.0, 0.0, fallback));
    const double norm = cv::norm(gravity);
    return (std::isfinite(norm) && norm > 0.0) ? norm : fallback;
}

inline bool TryReadPositiveVector3(const cv::FileStorage& fs,
                                   std::initializer_list<const char*> keys,
                                   Eigen::Vector3f& out)
{
    const cv::FileNode node = FindNode(fs, keys);
    if (node.empty()) {
        return false;
    }

    const cv::Vec3d value = ReadVec3(fs, keys);
    for (int i = 0; i < 3; ++i) {
        const double component = value[i];
        if (!std::isfinite(component) || component <= 0.0) {
            return false;
        }
        out(i) = static_cast<float>(component);
    }
    return true;
}

inline Eigen::Vector3f ReadAllanVector3(const cv::FileStorage& fs,
                                        const std::array<const char*, 3>& keys,
                                        int index,
                                        const Eigen::Vector3f& fallback,
                                        bool* found = nullptr)
{
    bool hasAny = false;
    Eigen::Vector3f out = fallback;
    for (int axis = 0; axis < 3; ++axis) {
        const cv::FileNode node = fs[keys[axis]];
        if (node.empty()) {
            continue;
        }
        hasAny = true;
        const double value = ReadAllanEntry(fs, keys[axis], index, fallback(axis));
        out(axis) = static_cast<float>(value);
    }

    if (found != nullptr) {
        *found = hasAny;
    }
    return out;
}

inline Eigen::Vector3f ScalarToVector3(float value)
{
    return Eigen::Vector3f::Constant(value);
}

inline float AverageVector3(const Eigen::Vector3f& value)
{
    return value.mean();
}

inline Eigen::Vector3f ReadImuSigmaGVec(const cv::FileStorage& fs, float fallback = 1e-4f)
{
    const Eigen::Vector3f fallbackVec = ScalarToVector3(fallback);

    Eigen::Vector3f explicitVector = fallbackVec;
    if (TryReadPositiveVector3(
            fs, {"imu.sigma_g_xyz", "IMU.sigma_g_xyz", "imu.sigma_g_vec"}, explicitVector)) {
        return explicitVector;
    }

    bool haveAllan = false;
    const Eigen::Vector3f allanVector = ReadAllanVector3(
        fs, {"imu.allangx", "imu.allangy", "imu.allangz"}, 0, fallbackVec, &haveAllan);
    if (haveAllan) {
        return allanVector;
    }

    const float scalar = ReadFloat(fs, {"imu.sigma_g", "IMU.sigma_g"}, -1.0f);
    if (scalar > 0.0f) {
        return ScalarToVector3(scalar);
    }

    return fallbackVec;
}

inline Eigen::Vector3f ReadImuSigmaWgVec(const cv::FileStorage& fs, float fallback = 1e-6f)
{
    const Eigen::Vector3f fallbackVec = ScalarToVector3(fallback);

    Eigen::Vector3f explicitVector = fallbackVec;
    if (TryReadPositiveVector3(
            fs, {"imu.sigma_wg_xyz", "IMU.sigma_wg_xyz", "imu.sigma_wg_vec"}, explicitVector)) {
        return explicitVector;
    }

    bool haveAllan = false;
    const Eigen::Vector3f allanVector = ReadAllanVector3(
        fs, {"imu.allangx", "imu.allangy", "imu.allangz"}, 2, fallbackVec, &haveAllan);
    if (haveAllan) {
        return allanVector;
    }

    const float scalar = ReadFloat(fs, {"imu.sigma_wg", "IMU.sigma_wg"}, -1.0f);
    if (scalar > 0.0f) {
        return ScalarToVector3(scalar);
    }

    return fallbackVec;
}

inline Eigen::Vector3f ReadImuSigmaAVec(const cv::FileStorage& fs, float fallback = 2e-3f)
{
    const Eigen::Vector3f fallbackVec = ScalarToVector3(fallback);

    Eigen::Vector3f explicitVector = fallbackVec;
    if (TryReadPositiveVector3(
            fs, {"imu.sigma_a_xyz", "IMU.sigma_a_xyz", "imu.sigma_a_vec"}, explicitVector)) {
        return explicitVector;
    }

    bool haveAllan = false;
    const Eigen::Vector3f allanVector = ReadAllanVector3(
        fs, {"imu.allanax", "imu.allanay", "imu.allanaz"}, 0, fallbackVec, &haveAllan);
    if (haveAllan) {
        return allanVector;
    }

    const float scalar = ReadFloat(fs, {"imu.sigma_a", "IMU.sigma_a"}, -1.0f);
    if (scalar > 0.0f) {
        return ScalarToVector3(scalar);
    }

    return fallbackVec;
}

inline Eigen::Vector3f ReadImuSigmaWaVec(const cv::FileStorage& fs, float fallback = 3e-3f)
{
    const Eigen::Vector3f fallbackVec = ScalarToVector3(fallback);

    Eigen::Vector3f explicitVector = fallbackVec;
    if (TryReadPositiveVector3(
            fs, {"imu.sigma_wa_xyz", "IMU.sigma_wa_xyz", "imu.sigma_wa_vec"}, explicitVector)) {
        return explicitVector;
    }

    bool haveAllan = false;
    const Eigen::Vector3f allanVector = ReadAllanVector3(
        fs, {"imu.allanax", "imu.allanay", "imu.allanaz"}, 2, fallbackVec, &haveAllan);
    if (haveAllan) {
        return allanVector;
    }

    const float scalar = ReadFloat(fs, {"imu.sigma_wa", "IMU.sigma_wa"}, -1.0f);
    if (scalar > 0.0f) {
        return ScalarToVector3(scalar);
    }

    return fallbackVec;
}

inline double ReadImuSigmaG(const cv::FileStorage& fs, double fallback = 1e-4)
{
    return static_cast<double>(AverageVector3(ReadImuSigmaGVec(fs, static_cast<float>(fallback))));
}

inline double ReadImuSigmaWg(const cv::FileStorage& fs, double fallback = 1e-6)
{
    return static_cast<double>(AverageVector3(ReadImuSigmaWgVec(fs, static_cast<float>(fallback))));
}

inline double ReadImuSigmaA(const cv::FileStorage& fs, double fallback = 2e-3)
{
    return static_cast<double>(AverageVector3(ReadImuSigmaAVec(fs, static_cast<float>(fallback))));
}

inline double ReadImuSigmaWa(const cv::FileStorage& fs, double fallback = 3e-3)
{
    return static_cast<double>(AverageVector3(ReadImuSigmaWaVec(fs, static_cast<float>(fallback))));
}

inline cv::Mat NormalizeTransform4x4(const cv::Mat& input)
{
    if (input.empty()) {
        return cv::Mat();
    }

    cv::Mat output = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat source;
    input.convertTo(source, CV_64F);

    if (source.rows >= 3 && source.cols >= 4) {
        source(cv::Range(0, 3), cv::Range(0, 4))
            .copyTo(output(cv::Range(0, 3), cv::Range(0, 4)));
        return output;
    }

    return cv::Mat();
}

inline cv::Mat InvertRigidTransform(const cv::Mat& input)
{
    cv::Mat T = NormalizeTransform4x4(input);
    if (T.empty()) {
        return cv::Mat();
    }

    cv::Mat R = T(cv::Range(0, 3), cv::Range(0, 3)).clone();
    cv::Mat t = T(cv::Range(0, 3), cv::Range(3, 4)).clone();
    cv::Mat R_inv = R.t();
    cv::Mat t_inv = -R_inv * t;

    cv::Mat output = cv::Mat::eye(4, 4, CV_64F);
    R_inv.copyTo(output(cv::Range(0, 3), cv::Range(0, 3)));
    t_inv.copyTo(output(cv::Range(0, 3), cv::Range(3, 4)));
    return output;
}

inline cv::Mat ReadBodyToCameraTransform(const cv::FileStorage& fs, bool useGroundTruth)
{
    cv::Mat transform;
    if (useGroundTruth) {
        transform = ReadMat(fs, {"imu.tocolor_gt", "imu.tocolor"});
    } else {
        transform = ReadMat(fs, {"imu.tocolor"});
    }
    if (!transform.empty()) {
        return InvertRigidTransform(transform);
    }

    if (useGroundTruth) {
        transform = ReadMat(fs, {"Camera.T_BC0_GT"});
    } else {
        transform = ReadMat(fs, {"Camera.T_BC0"});
    }
    transform = NormalizeTransform4x4(transform);
    if (!transform.empty()) {
        return transform;
    }

    return cv::Mat::eye(4, 4, CV_64F);
}

inline std::array<double, 5> ReadDistCoeffs(const cv::FileStorage& fs)
{
    cv::Mat dist = ReadMat(fs, {"cam.dist"});
    if (dist.empty()) {
        dist = ReadMat(fs, {"cam.D"});
    }
    if (!dist.empty()) {
        std::array<double, 5> out{0.0, 0.0, 0.0, 0.0, 0.0};
        dist = dist.reshape(1, 1);
        dist.convertTo(dist, CV_64F);
        const int count = std::min<int>(5, static_cast<int>(dist.total()));
        for (int i = 0; i < count; ++i) {
            out[static_cast<std::size_t>(i)] = dist.at<double>(0, i);
        }
        return out;
    }

    std::array<double, 5> out{
        ReadDouble(fs, {"Camera.k1"}, 0.0),
        ReadDouble(fs, {"Camera.k2"}, 0.0),
        ReadDouble(fs, {"Camera.p1"}, 0.0),
        ReadDouble(fs, {"Camera.p2"}, 0.0),
        ReadDouble(fs, {"Camera.k3"}, 0.0)
    };

    const bool haveOfficial =
        FindNode(fs, {"Camera.k1", "Camera.k2", "Camera.p1", "Camera.p2", "Camera.k3"}).empty() == false;
    if (haveOfficial) {
        return out;
    }
    return out;
}

} // namespace config_compat
} // namespace RVIO2

#endif
