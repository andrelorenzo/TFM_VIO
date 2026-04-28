#ifndef RVIO2_ROS_COMPAT_H
#define RVIO2_ROS_COMPAT_H

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace ros
{

class Time
{
public:
    Time() : mSec(0.0) {}
    explicit Time(double sec) : mSec(sec) {}

    static Time now()
    {
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        const double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(now).count();
        return Time(seconds);
    }

    double toSec() const { return mSec; }

private:
    double mSec;
};


class Duration
{
public:
    Duration() : mSec(0.0) {}
    explicit Duration(double sec) : mSec(sec) {}

    double toSec() const { return mSec; }

private:
    double mSec;
};


class Publisher
{
public:
    template<typename T>
    void publish(const T&) const
    {
    }
};


class NodeHandle
{
public:
    template<typename T>
    Publisher advertise(const std::string&, int) const
    {
        return Publisher();
    }
};


namespace detail
{

inline void Log(const char* level, const char* fmt, ...)
{
    std::fprintf(stderr, "[%s] ", level);

    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    std::fprintf(stderr, "\n");
}

} // namespace detail


namespace package
{

inline std::string getPath(const std::string&)
{
    return std::filesystem::current_path().string();
}

} // namespace package

} // namespace ros


#define ROS_INFO(...) do {} while (0)
#define ROS_ERROR(...) ::ros::detail::Log("ERROR", __VA_ARGS__)
#define ROS_DEBUG(...) do {} while (0)


namespace std_msgs
{

struct ColorRGBA
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;
};

} // namespace std_msgs


namespace geometry_msgs
{

struct Vector3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};


struct Quaternion
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
};


struct Point
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};


struct Pose
{
    Point position;
    Quaternion orientation;
};


struct Transform
{
    Vector3 translation;
    Quaternion rotation;
};


struct Header
{
    ros::Time stamp;
    std::string frame_id;
};


struct PoseStamped
{
    Header header;
    Pose pose;
};


struct TransformStamped
{
    Header header;
    std::string child_frame_id;
    Transform transform;
};

} // namespace geometry_msgs


namespace nav_msgs
{

struct Path
{
    geometry_msgs::Header header;
    std::vector<geometry_msgs::PoseStamped> poses;
};

} // namespace nav_msgs


namespace sensor_msgs
{

struct Image
{
};

} // namespace sensor_msgs


namespace visualization_msgs
{

struct Marker
{
    enum
    {
        ADD = 0,
        POINTS = 8
    };

    geometry_msgs::Header header;
    std::string ns;
    int id = 0;
    std_msgs::ColorRGBA color;
    geometry_msgs::Vector3 scale;
    geometry_msgs::Pose pose;
    ros::Duration lifetime;
    int action = ADD;
    int type = POINTS;
    std::vector<geometry_msgs::Point> points;
};

} // namespace visualization_msgs


namespace tf
{

class TransformBroadcaster
{
public:
    template<typename T>
    void sendTransform(const T&) const
    {
    }
};

} // namespace tf


namespace cv_bridge
{

struct CvImage
{
    std::string encoding;
    cv::Mat image;

    sensor_msgs::Image toImageMsg() const
    {
        return sensor_msgs::Image();
    }
};

} // namespace cv_bridge

#endif
