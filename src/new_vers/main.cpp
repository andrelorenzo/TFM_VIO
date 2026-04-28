#define LOGGER_IMP
#include "config.hpp"
#include "lie_math.hpp"
#include "plotter.hpp"
#include "pre_int.hpp"

#include "../rvio2/System.h"

#include "librealsense2/rs.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

Config MakePlotConfig();

enum class StreamSlot {
    Color = 0,
    Accel = 1,
    Gyro = 2,
    Count = 3
};

struct AppInputs {
    std::string bag_path;
    std::string rvio_config_path = "src/rvio2.yaml";
    Config plot_config{};
    double target_color_fps = 0.0;
};

struct DebugCounters {
    size_t accel_samples = 0;
    size_t gyro_samples = 0;
    size_t color_frames_seen = 0;
    size_t color_frames_kept = 0;
    size_t packets_created = 0;
    size_t packets_popped = 0;
    size_t packets_without_imu = 0;
    size_t packets_without_pose = 0;
    size_t poses_out = 0;
    size_t frames_dropped_before_first_imu = 0;
};

DebugCounters g_debug;

constexpr bool kVerboseSensorLogs = false;
constexpr bool kVerboseBridgeLogs = false;

class TimestampNormalizer {
public:
    double Normalize(double raw_ms, StreamSlot slot) {
        if (origin_ms_ < 0.0) {
            origin_ms_ = raw_ms;
        }

        const int idx = static_cast<int>(slot);
        if (last_raw_ms_[idx] >= 0.0 && raw_ms + 100.0 < last_raw_ms_[idx]) {
            wrap_ms_ += (last_norm_ms_[idx] - (raw_ms + wrap_ms_ - origin_ms_)) + 1.0;
        }

        double norm_ms = raw_ms + wrap_ms_ - origin_ms_;
        if (last_norm_ms_[idx] >= 0.0 && norm_ms < last_norm_ms_[idx]) {
            norm_ms = last_norm_ms_[idx];
        }

        last_raw_ms_[idx] = raw_ms;
        last_norm_ms_[idx] = norm_ms;
        return norm_ms;
    }

private:
    double origin_ms_ = -1.0;
    double wrap_ms_ = 0.0;
    double last_raw_ms_[static_cast<int>(StreamSlot::Count)] = {-1.0, -1.0, -1.0};
    double last_norm_ms_[static_cast<int>(StreamSlot::Count)] = {-1.0, -1.0, -1.0};
};

class Rvio2Bridge {
public:
    explicit Rvio2Bridge(const std::string& config_path)
        : system_(config_path) {
    }

    bool PushPacket(const SourceIn& raw_packet, RVIO2::PoseEstimate* out_pose) {
        SourceIn packet = raw_packet;
        bool imu_ready = false;
        if (!packet.gyr.empty() && !packet.acc.empty() &&
            !packet.gyr_tsms.empty() && !packet.acc_tsms.empty()) {
            imu_ready = ResampleAccToGyroInPlace(&packet);
        }

        if (!imu_ready) {
            ++g_debug.packets_without_imu;
            if (kVerboseBridgeLogs && (g_debug.packets_without_imu <= 10 || (g_debug.packets_without_imu % 25) == 0)) {
                std::cout << "[RVIO2] paquete sin IMU util: frame_ts="
                          << raw_packet.frame_tsms
                          << " acc=" << raw_packet.acc.size()
                          << " gyr=" << raw_packet.gyr.size()
                          << std::endl;
            }
        }

        const size_t imu_count = imu_ready ? std::min(packet.gyr.size(), packet.acc.size()) : 0;
        for (size_t i = 0; i < imu_count; ++i) {
            const double timestamp_sec = packet.gyr_tsms[i] * 1e-3;
            if (first_imu_timestamp_sec_ < 0.0) {
                first_imu_timestamp_sec_ = timestamp_sec;
            }
            if (last_imu_timestamp_sec_ >= 0.0 && timestamp_sec <= last_imu_timestamp_sec_ + 1e-9) {
                continue;
            }

            RVIO2::ImuData* data = new RVIO2::ImuData();
            data->Timestamp = timestamp_sec;
            data->TimeInterval = last_imu_timestamp_sec_ < 0.0 ? 0.0 : (timestamp_sec - last_imu_timestamp_sec_);
            data->AngularVel << static_cast<float>(packet.gyr[i].x()),
                                static_cast<float>(packet.gyr[i].y()),
                                static_cast<float>(packet.gyr[i].z());
            data->LinearAccel << static_cast<float>(packet.acc[i].x()),
                                 static_cast<float>(packet.acc[i].y()),
                                 static_cast<float>(packet.acc[i].z());

            last_imu_timestamp_sec_ = timestamp_sec;
            system_.PushImuData(data);
        }

        if (packet.frame.empty()) {
            return false;
        }

        const double frame_timestamp_sec = packet.frame_tsms * 1e-3;
        if (first_imu_timestamp_sec_ < 0.0) {
            return false;
        }
        if (frame_timestamp_sec <= first_imu_timestamp_sec_ + 1e-9) {
            ++g_debug.frames_dropped_before_first_imu;
            if (kVerboseBridgeLogs && (g_debug.frames_dropped_before_first_imu <= 10 ||
                (g_debug.frames_dropped_before_first_imu % 25) == 0)) {
                std::cout << "[RVIO2] frame descartado: llega antes del primer IMU util"
                          << " frame_ts=" << packet.frame_tsms
                          << " first_imu_ts=" << (first_imu_timestamp_sec_ * 1e3)
                          << std::endl;
            }
            return false;
        }

        RVIO2::ImageData* image = new RVIO2::ImageData();
        image->Timestamp = frame_timestamp_sec;
        if (packet.frame.channels() == 1) {
            image->Image = packet.frame.clone();
        } else if (packet.frame.channels() == 3) {
            cv::cvtColor(packet.frame, image->Image, cv::COLOR_BGR2GRAY);
        } else if (packet.frame.channels() == 4) {
            cv::cvtColor(packet.frame, image->Image, cv::COLOR_BGRA2GRAY);
        } else {
            image->Image = packet.frame.clone();
        }
        system_.PushImageData(image);

        RVIO2::PoseEstimate pose;
        bool has_pose = false;
        while (system_.run(&pose)) {
            latest_pose_ = pose;
            has_latest_pose_ = true;
            has_pose = true;
        }

        if (has_pose && out_pose != nullptr) {
            *out_pose = latest_pose_;
        }
        if (!has_pose) {
            ++g_debug.packets_without_pose;
            if (kVerboseBridgeLogs && (g_debug.packets_without_pose <= 10 || (g_debug.packets_without_pose % 25) == 0)) {
                std::cout << "[RVIO2] run() aun sin pose: frame_ts=" << packet.frame_tsms
                          << " imu_resampled=" << imu_count
                          << std::endl;
            }
        }
        return has_pose;
    }

private:
    double last_imu_timestamp_sec_ = -1.0;
    double first_imu_timestamp_sec_ = -1.0;
    bool has_latest_pose_ = false;
    RVIO2::PoseEstimate latest_pose_;
    RVIO2::System system_;
};

class ImuFramePacketizer {
public:
    explicit ImuFramePacketizer(double target_color_fps)
        : min_frame_dt_ms_(target_color_fps > 0.0 ? (1000.0 / target_color_fps) : 0.0) {
    }

    void PushAccel(double ts_ms, const vec3& sample) {
        PushSample(&accel_accum_, &accel_tsms_accum_, &accel_dtms_accum_, ts_ms, sample);
        ++g_debug.accel_samples;
        if (kVerboseSensorLogs && (g_debug.accel_samples <= 8 || (g_debug.accel_samples % 200) == 0)) {
            std::cout << "[ACC] ts=" << ts_ms
                      << " a=[" << sample.x() << ", " << sample.y() << ", " << sample.z() << "]"
                      << " cached=" << accel_accum_.size()
                      << std::endl;
        }
    }

    void PushGyro(double ts_ms, const vec3& sample) {
        PushSample(&gyro_accum_, &gyro_tsms_accum_, &gyro_dtms_accum_, ts_ms, sample);
        ++g_debug.gyro_samples;
        if (kVerboseSensorLogs && (g_debug.gyro_samples <= 8 || (g_debug.gyro_samples % 200) == 0)) {
            std::cout << "[GYR] ts=" << ts_ms
                      << " w=[" << sample.x() << ", " << sample.y() << ", " << sample.z() << "]"
                      << " cached=" << gyro_accum_.size()
                      << std::endl;
        }
    }

    void PushColorFrame(double ts_ms, const cv::Mat& image_bgr) {
        if (image_bgr.empty()) {
            return;
        }
        ++g_debug.color_frames_seen;
        if (has_last_frame_seen_ && ts_ms <= last_frame_seen_ts_ms_ + 1e-9) {
            return;
        }
        if (min_frame_dt_ms_ > 0.0 &&
            has_last_frame_seen_ &&
            (ts_ms - last_frame_seen_ts_ms_) < min_frame_dt_ms_ - 1e-9) {
            return;
        }

        SourceIn packet;
        packet.frame = image_bgr.clone();
        packet.frame_tsms = ts_ms;
        packet.frame_dtms = has_last_frame_seen_ ? (ts_ms - last_frame_seen_ts_ms_) : 0.0;
        packet.depth.release();
        packet.depth_tsms = 0.0;

        SplitSamplesForFrame(ts_ms,
                             accel_accum_, accel_tsms_accum_,
                             &packet.acc, &packet.acc_tsms, &packet.acc_dtms,
                             &accel_accum_, &accel_tsms_accum_, &accel_dtms_accum_);
        SplitSamplesForFrame(ts_ms,
                             gyro_accum_, gyro_tsms_accum_,
                             &packet.gyr, &packet.gyr_tsms, &packet.gyr_dtms,
                             &gyro_accum_, &gyro_tsms_accum_, &gyro_dtms_accum_);

        ready_packets_.push_back(std::move(packet));
        last_frame_seen_ts_ms_ = ts_ms;
        has_last_frame_seen_ = true;
        ++g_debug.color_frames_kept;
        ++g_debug.packets_created;
        if (kVerboseSensorLogs && (g_debug.color_frames_kept <= 8 || (g_debug.color_frames_kept % 25) == 0)) {
            const SourceIn& dbg_pkt = ready_packets_.back();
            std::cout << "[FRAME] ts=" << ts_ms
                      << " dt=" << dbg_pkt.frame_dtms
                      << " acc=" << dbg_pkt.acc.size();
            if (!dbg_pkt.acc_tsms.empty()) {
                std::cout << " acc_t=[" << dbg_pkt.acc_tsms.front() << "," << dbg_pkt.acc_tsms.back() << "]";
            }
            std::cout << " gyr=" << dbg_pkt.gyr.size();
            if (!dbg_pkt.gyr_tsms.empty()) {
                std::cout << " gyr_t=[" << dbg_pkt.gyr_tsms.front() << "," << dbg_pkt.gyr_tsms.back() << "]";
            }
            std::cout << " q=" << ready_packets_.size()
                      << std::endl;
        }
    }

    bool PopReadyPacket(SourceIn* packet) {
        if (packet == nullptr || ready_packets_.empty()) {
            return false;
        }
        *packet = std::move(ready_packets_.front());
        ready_packets_.pop_front();
        ++g_debug.packets_popped;
        if (kVerboseSensorLogs && (g_debug.packets_popped <= 8 || (g_debug.packets_popped % 25) == 0)) {
            std::cout << "[PKT] pop ts=" << packet->frame_tsms
                      << " acc=" << packet->acc.size()
                      << " gyr=" << packet->gyr.size()
                      << " q=" << ready_packets_.size()
                      << std::endl;
        }
        return true;
    }

private:
    static void PushSample(std::deque<vec3>* samples,
                           std::deque<double>* tsms,
                           std::deque<double>* dtms,
                           double ts_ms,
                           const vec3& sample) {
        if (samples == nullptr || tsms == nullptr || dtms == nullptr) {
            return;
        }
        if (!tsms->empty() && ts_ms <= tsms->back() + 1e-9) {
            if (std::abs(ts_ms - tsms->back()) < 1e-9 && !samples->empty()) {
                samples->back() = sample;
            }
            return;
        }

        if (!tsms->empty()) {
            dtms->push_back(ts_ms - tsms->back());
        }
        samples->push_back(sample);
        tsms->push_back(ts_ms);
    }

    static void RebuildDtms(const std::deque<double>& timestamps_ms, std::deque<double>* dtms) {
        if (dtms == nullptr) {
            return;
        }

        dtms->clear();
        for (size_t i = 1; i < timestamps_ms.size(); ++i) {
            dtms->emplace_back(timestamps_ms[i] - timestamps_ms[i - 1]);
        }
    }

    static void SplitSamplesForFrame(double frame_ts_ms,
                                     const std::deque<vec3>& src_data,
                                     const std::deque<double>& src_tsms,
                                     std::deque<vec3>* packet_data,
                                     std::deque<double>* packet_tsms,
                                     std::deque<double>* packet_dtms,
                                     std::deque<vec3>* remain_data,
                                     std::deque<double>* remain_tsms,
                                     std::deque<double>* remain_dtms) {
        if (packet_data == nullptr || packet_tsms == nullptr || packet_dtms == nullptr ||
            remain_data == nullptr || remain_tsms == nullptr || remain_dtms == nullptr) {
            return;
        }

        const std::deque<vec3> src_data_copy = src_data;
        const std::deque<double> src_tsms_copy = src_tsms;

        packet_data->clear();
        packet_tsms->clear();
        packet_dtms->clear();
        remain_data->clear();
        remain_tsms->clear();
        remain_dtms->clear();

        if (src_data_copy.size() != src_tsms_copy.size()) {
            return;
        }

        size_t first_future = 0;
        while (first_future < src_tsms_copy.size() && src_tsms_copy[first_future] <= frame_ts_ms + 1e-9) {
            ++first_future;
        }

        for (size_t i = 0; i < first_future; ++i) {
            packet_data->push_back(src_data_copy[i]);
            packet_tsms->push_back(src_tsms_copy[i]);
        }
        RebuildDtms(*packet_tsms, packet_dtms);

        if (first_future == 0) {
            *remain_data = src_data_copy;
            *remain_tsms = src_tsms_copy;
            RebuildDtms(*remain_tsms, remain_dtms);
            return;
        }

        remain_data->push_back(src_data_copy[first_future - 1]);
        remain_tsms->push_back(src_tsms_copy[first_future - 1]);
        for (size_t i = first_future; i < src_tsms_copy.size(); ++i) {
            remain_data->push_back(src_data_copy[i]);
            remain_tsms->push_back(src_tsms_copy[i]);
        }
        RebuildDtms(*remain_tsms, remain_dtms);
    }

private:
    std::deque<vec3> accel_accum_;
    std::deque<double> accel_tsms_accum_;
    std::deque<double> accel_dtms_accum_;
    std::deque<vec3> gyro_accum_;
    std::deque<double> gyro_tsms_accum_;
    std::deque<double> gyro_dtms_accum_;
    std::deque<SourceIn> ready_packets_;
    double min_frame_dt_ms_ = 0.0;
    double last_frame_seen_ts_ms_ = -1.0;
    bool has_last_frame_seen_ = false;
};

Config MakePlotConfig() {
    Config config{};
    config.gen.plot_tray = true;
    config.gen.plot_2d = false;
    config.gen.plot_vis_tray = false;
    config.gen.plot_imu_tray = false;
    config.gen.plot_gt_with_tray = false;
    config.gen.plot_gt_with_vis_tray = false;
    config.gen.plot_gt_with_imu_tray = false;
    config.gen.plot_imu = false;
    config.gen.plot_rpy = false;
    config.gen.plot_vis_rpy = false;
    config.gen.plot_gt_with_rpy = false;
    config.gen.plot_gt_with_vis_rpy = false;
    return config;
}

bool WantAnyPlots(const Config& cfg) {
    return cfg.gen.plot_tray ||
           cfg.gen.plot_vis_tray ||
           cfg.gen.plot_imu_tray ||
           cfg.gen.plot_gt_with_tray ||
           cfg.gen.plot_gt_with_vis_tray ||
           cfg.gen.plot_gt_with_imu_tray ||
           cfg.gen.plot_imu ||
           cfg.gen.plot_rpy ||
           cfg.gen.plot_vis_rpy ||
           cfg.gen.plot_gt_with_rpy ||
           cfg.gen.plot_gt_with_vis_rpy;
}

double ReadRvioCameraFps(const std::string& rvio_config_path) {
    cv::FileStorage fs(rvio_config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return 0.0;
    }

    cv::FileNode node = fs["Camera.fps"];
    if (node.empty()) {
        return 0.0;
    }

    double fps = 0.0;
    node >> fps;
    return fps > 0.0 ? fps : 0.0;
}

std::string TimestampDomainToString(rs2_timestamp_domain domain) {
    switch (domain) {
        case RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK:
            return "HARDWARE_CLOCK";
        case RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME:
            return "SYSTEM_TIME";
        case RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME:
            return "GLOBAL_TIME";
        default:
            return "UNKNOWN";
    }
}

std::string ToLowerCopy(std::string text) {
    std::transform(
        text.begin(),
        text.end(),
        text.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

bool EndsWithNoCase(const std::string& text, const std::string& suffix) {
    const std::string text_lc = ToLowerCopy(text);
    const std::string suffix_lc = ToLowerCopy(suffix);
    if (suffix_lc.size() > text_lc.size()) {
        return false;
    }
    return text_lc.compare(text_lc.size() - suffix_lc.size(), suffix_lc.size(), suffix_lc) == 0;
}

std::string ResolvePathRelativeToFile(const std::string& file_path, const std::string& target_path) {
    if (target_path.empty()) {
        return target_path;
    }

    const std::filesystem::path target(target_path);
    if (target.is_absolute()) {
        return target.lexically_normal().string();
    }

    const std::filesystem::path base(file_path);
    if (base.has_parent_path()) {
        return (base.parent_path() / target).lexically_normal().string();
    }

    return target.lexically_normal().string();
}

AppInputs ParseAppInputs(int argc, char** argv) {
    if (argc < 2) {
        throw std::runtime_error(
            "Uso: new_vers <path-to-bag | path-to-config.yaml> [path-to-rvio2.yaml]");
    }

    AppInputs inputs;
    inputs.plot_config = MakePlotConfig();

    const std::string source_arg = argv[1];
    if (argc > 2) {
        inputs.rvio_config_path = argv[2];
    }
    inputs.target_color_fps = ReadRvioCameraFps(inputs.rvio_config_path);

    if (EndsWithNoCase(source_arg, ".bag")) {
        inputs.plot_config.gen.plot_tray = true;
        inputs.bag_path = source_arg;
        return inputs;
    }

    if (EndsWithNoCase(source_arg, ".yaml") || EndsWithNoCase(source_arg, ".yml")) {
        Config parsed_config{};
        if (!parsed_config.parseYAML(source_arg.c_str())) {
            throw std::runtime_error("No se pudo abrir o parsear el config YAML: " + source_arg);
        }

        const std::string resolved_bag_path =
            ResolvePathRelativeToFile(source_arg, parsed_config.gen.input);

        if (!EndsWithNoCase(resolved_bag_path, ".bag")) {
            throw std::runtime_error(
                "El config YAML debe apuntar a un fichero .bag en gen.input");
        }

        inputs.bag_path = resolved_bag_path;
        inputs.plot_config = parsed_config;
        inputs.plot_config.gen.plot_tray = true;
        if (parsed_config.cam.fps > 0.0) {
            inputs.target_color_fps = parsed_config.cam.fps;
        }
        return inputs;
    }

    throw std::runtime_error(
        "La primera entrada debe ser un fichero .bag o un config .yaml/.yml");
}

cv::Mat ConvertColorFrameToBgr(const rs2::video_frame& frame) {
    if (!frame) {
        return cv::Mat();
    }

    const cv::Size size(frame.get_width(), frame.get_height());
    const rs2_format format = frame.get_profile().format();
    void* data = const_cast<void*>(frame.get_data());

    switch (format) {
        case RS2_FORMAT_BGR8: {
            cv::Mat image(size, CV_8UC3, data, cv::Mat::AUTO_STEP);
            return image.clone();
        }
        case RS2_FORMAT_RGB8: {
            cv::Mat rgb(size, CV_8UC3, data, cv::Mat::AUTO_STEP);
            cv::Mat bgr;
            cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
            return bgr;
        }
        case RS2_FORMAT_BGRA8: {
            cv::Mat bgra(size, CV_8UC4, data, cv::Mat::AUTO_STEP);
            cv::Mat bgr;
            cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
            return bgr;
        }
        case RS2_FORMAT_RGBA8: {
            cv::Mat rgba(size, CV_8UC4, data, cv::Mat::AUTO_STEP);
            cv::Mat bgr;
            cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
            return bgr;
        }
        case RS2_FORMAT_YUYV: {
            cv::Mat yuyv(size, CV_8UC2, data, cv::Mat::AUTO_STEP);
            cv::Mat bgr;
            cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2);
            return bgr;
        }
        case RS2_FORMAT_UYVY: {
            cv::Mat uyvy(size, CV_8UC2, data, cv::Mat::AUTO_STEP);
            cv::Mat bgr;
            cv::cvtColor(uyvy, bgr, cv::COLOR_YUV2BGR_UYVY);
            return bgr;
        }
        case RS2_FORMAT_Y8: {
            cv::Mat gray(size, CV_8UC1, data, cv::Mat::AUTO_STEP);
            cv::Mat bgr;
            cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
            return bgr;
        }
        case RS2_FORMAT_MJPEG: {
            const unsigned char* begin = static_cast<const unsigned char*>(frame.get_data());
            const unsigned char* end = begin + frame.get_data_size();
            std::vector<unsigned char> compressed(begin, end);
            return cv::imdecode(compressed, cv::IMREAD_COLOR);
        }
        default: {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::cerr << "Formato de color no soportado por ConvertColorFrameToBgr: "
                          << rs2_format_to_string(format) << std::endl;
            }
            return cv::Mat();
        }
    }
}

void DrainReadyPackets(ImuFramePacketizer* packetizer,
                       Rvio2Bridge* bridge,
                       StateOut* state,
                       double* last_pose_ts_sec,
                       vec3* last_pose_position,
                       std::vector<StateOut>* recorded_states = nullptr,
                       bool live_plot = false,
                       bool live_print = false) {
    if (packetizer == nullptr || bridge == nullptr || state == nullptr ||
        last_pose_ts_sec == nullptr || last_pose_position == nullptr) {
        return;
    }

    SourceIn packet;
    while (packetizer->PopReadyPacket(&packet)) {
        RVIO2::PoseEstimate pose;
        if (!bridge->PushPacket(packet, &pose)) {
            continue;
        }

        const vec3 position(static_cast<double>(pose.Position(0)),
                            static_cast<double>(pose.Position(1)),
                            static_cast<double>(pose.Position(2)));
        const quat orientation = normalizeQ(
            quat(static_cast<double>(pose.Quaternion(3)),
                 static_cast<double>(pose.Quaternion(0)),
                 static_cast<double>(pose.Quaternion(1)),
                 static_cast<double>(pose.Quaternion(2))));
        ++g_debug.poses_out;

        double dt_sec = 0.0;
        if (*last_pose_ts_sec >= 0.0) {
            dt_sec = std::max(0.0, pose.Timestamp - *last_pose_ts_sec);
        }

        state->ts_ms = pose.Timestamp * 1000.0;
        state->dt = dt_sec;
        state->pos_m = position;
        state->quat_rad = orientation;
        state->rpy_rad = quatToCameraRpyRad(orientation);
        if (dt_sec > 1e-6) {
            state->vel_ms = (position - *last_pose_position) / dt_sec;
        } else {
            state->vel_ms = vec3::Zero();
        }

        if (!packet.acc.empty()) {
            state->acc_cal_ms2 = packet.acc.back();
            state->deb.acc_ms2 = packet.acc.back();
        }
        if (!packet.gyr.empty()) {
            state->gyr_cal_rads = packet.gyr.back();
            state->deb.gyr_rads = packet.gyr.back();
        }

        state->deb.vis_xyz = position;
        state->deb.vis_rpy = state->rpy_rad;
        state->deb.imu_xyz = position;
        state->deb.imu_rpy = state->rpy_rad;
        state->deb.scale = 1.0f;
        state->deb.vio_valid = true;

        *last_pose_ts_sec = pose.Timestamp;
        *last_pose_position = position;

        if (recorded_states != nullptr) {
            recorded_states->push_back(*state);
        }
        if (live_plot) {
            updatePlots(state);
        }
        if (live_print) {
            std::cout << "[POSE] t=" << pose.Timestamp
                      << "  p=[" << position.x() << ", " << position.y() << ", " << position.z() << "]"
                      << std::endl;
        }
    }
}

void ProcessFrame(const rs2::frame& frame,
                  TimestampNormalizer* time_normalizer,
                  ImuFramePacketizer* packetizer) {
    if (!frame || time_normalizer == nullptr || packetizer == nullptr) {
        return;
    }

    if (rs2::frameset frames = frame.as<rs2::frameset>()) {
        std::vector<rs2::frame> motion_frames;
        std::vector<rs2::frame> color_frames;

        for (const rs2::frame& subframe : frames) {
            const rs2_stream stream = subframe.get_profile().stream_type();
            if (stream == RS2_STREAM_ACCEL || stream == RS2_STREAM_GYRO) {
                motion_frames.push_back(subframe);
            } else if (stream == RS2_STREAM_COLOR) {
                color_frames.push_back(subframe);
            }
        }

        for (const rs2::frame& motion : motion_frames) {
            ProcessFrame(motion, time_normalizer, packetizer);
        }
        for (const rs2::frame& color : color_frames) {
            ProcessFrame(color, time_normalizer, packetizer);
        }
        return;
    }

    const rs2_stream stream = frame.get_profile().stream_type();
    if (stream == RS2_STREAM_ACCEL) {
        rs2::motion_frame motion = frame.as<rs2::motion_frame>();
        if (!motion) {
            return;
        }
        const rs2_vector v = motion.get_motion_data();
        const double ts_ms = time_normalizer->Normalize(motion.get_timestamp(), StreamSlot::Accel);
        static int accel_debug = 0;
        if (kVerboseSensorLogs && accel_debug < 3) {
            ++accel_debug;
            std::cout << "[ACC_RAW] ts_raw=" << motion.get_timestamp()
                      << " ts_norm=" << ts_ms
                      << " domain=" << TimestampDomainToString(motion.get_frame_timestamp_domain())
                      << std::endl;
        }
        packetizer->PushAccel(ts_ms, vec3(v.x, v.y, v.z));
        return;
    }

    if (stream == RS2_STREAM_GYRO) {
        rs2::motion_frame motion = frame.as<rs2::motion_frame>();
        if (!motion) {
            return;
        }
        const rs2_vector v = motion.get_motion_data();
        const double ts_ms = time_normalizer->Normalize(motion.get_timestamp(), StreamSlot::Gyro);
        static int gyro_debug = 0;
        if (kVerboseSensorLogs && gyro_debug < 3) {
            ++gyro_debug;
            std::cout << "[GYR_RAW] ts_raw=" << motion.get_timestamp()
                      << " ts_norm=" << ts_ms
                      << " domain=" << TimestampDomainToString(motion.get_frame_timestamp_domain())
                      << std::endl;
        }
        packetizer->PushGyro(ts_ms, vec3(v.x, v.y, v.z));
        return;
    }

    if (stream == RS2_STREAM_COLOR) {
        rs2::video_frame color = frame.as<rs2::video_frame>();
        if (!color) {
            return;
        }
        const double ts_ms = time_normalizer->Normalize(color.get_timestamp(), StreamSlot::Color);
        static int color_debug = 0;
        if (kVerboseSensorLogs && color_debug < 5) {
            ++color_debug;
            std::cout << "[COLOR_RAW] ts_raw=" << color.get_timestamp()
                      << " ts_norm=" << ts_ms
                      << " domain=" << TimestampDomainToString(color.get_frame_timestamp_domain())
                      << " fmt=" << rs2_format_to_string(color.get_profile().format())
                      << " size=" << color.get_width() << "x" << color.get_height()
                      << std::endl;
        }
        const cv::Mat image_bgr = ConvertColorFrameToBgr(color);
        if (!image_bgr.empty()) {
            packetizer->PushColorFrame(ts_ms, image_bgr);
        }
    }
}

void PrintProfileSummary(const rs2::pipeline_profile& profile) {
    std::cout << "Streams activos:" << std::endl;
    for (const rs2::stream_profile& stream : profile.get_streams()) {
        const rs2_stream type = stream.stream_type();
        std::cout << "  - " << rs2_stream_to_string(type)
                  << " @ " << stream.fps() << " Hz"
                  << " [" << rs2_format_to_string(stream.format()) << "]";

        if (rs2::video_stream_profile video = stream.as<rs2::video_stream_profile>()) {
            std::cout << " (" << video.width() << "x" << video.height() << ")";
        }

        std::cout << std::endl;
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        AppInputs inputs = ParseAppInputs(argc, argv);

        StateOut state{};
        initPlotters(&inputs.plot_config);

        Rvio2Bridge bridge(inputs.rvio_config_path);
        TimestampNormalizer time_normalizer;
        ImuFramePacketizer packetizer(inputs.target_color_fps);

        rs2::pipeline pipe;
        rs2::config rs_config;
        rs_config.enable_device_from_file(inputs.bag_path, false);
        rs_config.enable_stream(RS2_STREAM_COLOR, -1, 0, 0, RS2_FORMAT_RGB8, 0);
        rs_config.enable_stream(RS2_STREAM_ACCEL);
        rs_config.enable_stream(RS2_STREAM_GYRO);

        rs2::frame_queue queue(2048);
        std::vector<StateOut> recorded_states;
        recorded_states.reserve(4096);
        bool live_plot = false;
        bool live_print = false;

        rs2::pipeline_profile profile = pipe.start(
            rs_config,
            [&](const rs2::frame& frame) {
                queue.enqueue(std::move(frame));
            });

        if (!profile.get_device().is<rs2::playback>()) {
            throw std::runtime_error("La fuente abierta no es un playback de RealSense");
        }

        rs2::playback playback = profile.get_device().as<rs2::playback>();
#ifdef _WIN32
        playback.set_real_time(false);
        std::cout << "[INFO] En Windows se usa playback real-time con cola ligera; el procesado pesado se hace fuera del callback para evitar los cuelgues de set_real_time(false)." << std::endl;
#else
        // playback.set_real_time(false);
        std::cout << "[INFO] Playback en modo no real-time." << std::endl;
#endif

        std::atomic<bool> playback_started{false};
        std::atomic<bool> playback_stopped{false};
        playback.set_status_changed_callback(
            [&](rs2_playback_status status) {
                if (status == RS2_PLAYBACK_STATUS_PLAYING || status == RS2_PLAYBACK_STATUS_PAUSED) {
                    playback_started.store(true, std::memory_order_relaxed);
                }
                if (status == RS2_PLAYBACK_STATUS_STOPPED &&
                    playback_started.load(std::memory_order_relaxed)) {
                    playback_stopped.store(true, std::memory_order_relaxed);
                }
            });

        PrintProfileSummary(profile);
        std::cout << "Usando bag: " << inputs.bag_path << std::endl;
        std::cout << "Usando configuracion RVIO2: " << inputs.rvio_config_path << std::endl;
        std::cout << "FPS objetivo de color: " << inputs.target_color_fps << std::endl;
        std::cout << "Salida en vivo: se acumulan poses y se dibujan al final para no perturbar el playback." << std::endl;

        double bag_color_fps = 0.0;
        for (const rs2::stream_profile& stream : profile.get_streams()) {
            if (stream.stream_type() == RS2_STREAM_COLOR) {
                bag_color_fps = static_cast<double>(stream.fps());
                break;
            }
        }
        if (inputs.target_color_fps > 0.0 && bag_color_fps > inputs.target_color_fps + 1e-3) {
            std::cout << "[AVISO] El bag trae color a " << bag_color_fps
                      << " Hz, pero el bridge esta filtrando a " << inputs.target_color_fps
                      << " Hz para igualar rvio2.yaml. En secuencias cortas o con poca paralaje esto puede impedir la inicializacion."
                      << std::endl;
        }

        double last_pose_ts_sec = -1.0;
        vec3 last_pose_position = vec3::Zero();
        bool saw_any_frame = false;
        int idle_loops_after_stop = 0;

        while (true) {
            rs2::frame frame;
            if (queue.poll_for_frame(&frame)) {
                saw_any_frame = true;
                playback_started.store(true, std::memory_order_relaxed);
                idle_loops_after_stop = 0;
                ProcessFrame(frame, &time_normalizer, &packetizer);
                DrainReadyPackets(
                    &packetizer,
                    &bridge,
                    &state,
                    &last_pose_ts_sec,
                    &last_pose_position,
                    &recorded_states,
                    live_plot,
                    live_print);
                continue;
            }

            DrainReadyPackets(
                &packetizer,
                &bridge,
                &state,
                &last_pose_ts_sec,
                &last_pose_position,
                &recorded_states,
                live_plot,
                live_print);

            if (saw_any_frame && playback_stopped.load(std::memory_order_relaxed)) {
                ++idle_loops_after_stop;
                if (idle_loops_after_stop > 300) {
                    break;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        DrainReadyPackets(
            &packetizer,
            &bridge,
            &state,
            &last_pose_ts_sec,
            &last_pose_position,
            &recorded_states,
            live_plot,
            live_print);

        std::cout << "Playback terminado." << std::endl;
        std::cout << "[RESUMEN] acc=" << g_debug.accel_samples
                  << " gyr=" << g_debug.gyro_samples
                  << " frames_seen=" << g_debug.color_frames_seen
                  << " frames_kept=" << g_debug.color_frames_kept
                  << " pkts=" << g_debug.packets_created
                  << " poses=" << g_debug.poses_out
                  << " drop_pre_imu=" << g_debug.frames_dropped_before_first_imu
                  << " no_imu=" << g_debug.packets_without_imu
                  << " no_pose=" << g_debug.packets_without_pose
                  << std::endl;

        if (!recorded_states.empty()) {
            std::cout << "[POSE_FIRST] t=" << (recorded_states.front().ts_ms * 1e-3)
                      << "  p=[" << recorded_states.front().pos_m.x() << ", "
                      << recorded_states.front().pos_m.y() << ", "
                      << recorded_states.front().pos_m.z() << "]" << std::endl;
            std::cout << "[POSE_LAST]  t=" << (recorded_states.back().ts_ms * 1e-3)
                      << "  p=[" << recorded_states.back().pos_m.x() << ", "
                      << recorded_states.back().pos_m.y() << ", "
                      << recorded_states.back().pos_m.z() << "]" << std::endl;
        }

        if (WantAnyPlots(inputs.plot_config) && !recorded_states.empty()) {
            std::cout << "Renderizando grafica final con " << recorded_states.size() << " poses..." << std::endl;
            for (const StateOut& st_const : recorded_states) {
                StateOut st = st_const;
                updatePlots(&st);
            }
        }
        pipe.stop();
        closePlotters();
        return 0;
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what()
                  << " (" << e.get_failed_function() << ")"
                  << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    closePlotters();
    return 1;
}
