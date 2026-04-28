#include "source_man.hpp"
#include "seconds/comms_common.h"

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{
    struct StampedVec3
    {
        double ts = 0.0;
        cv::Vec3d value = {0.0, 0.0, 0.0};
        bool valid = false;
    };

    struct PendingGyroSample
    {
        double ts = 0.0;
        cv::Vec3d gyro = {0.0, 0.0, 0.0};
    };

    Config g_config;
    bool g_initialized = false;
    bool g_need_depth = false;

    rs2::pipeline g_pipe;
    rs2::config g_rs_cfg;
    rs2::pipeline_profile g_profile;
    bool g_rs_started = false;

    cv::VideoCapture g_cap;
    bool g_cap_started = false;

    std::ifstream g_csv;
    bool g_csv_started = false;
    bool g_csv_has_header = false;
    bool g_csv_has_timestamp = false;
    int g_csv_ts_col = -1;
    int g_csv_gx_col = -1;
    int g_csv_gy_col = -1;
    int g_csv_gz_col = -1;
    int g_csv_ax_col = -1;
    int g_csv_ay_col = -1;
    int g_csv_az_col = -1;
    std::uint64_t g_csv_row_index = 0;
    double g_csv_next_ts = 0.0;

    cv::Mat g_last_frame;
    cv::Mat g_last_depth_m;
    imuData g_last_imu;
    double g_last_gyro_ts = 0.0;
    double g_last_accel_ts = 0.0;
    double g_last_emitted_imu_ts = -std::numeric_limits<double>::infinity();
    bool g_have_gyro = false;
    bool g_have_accel = false;
    StampedVec3 g_prev_accel_sample;
    StampedVec3 g_curr_accel_sample;
    std::deque<PendingGyroSample> g_pending_gyro;

    double g_last_color_ts_ms = 0.0;
    double g_last_depth_ts_ms = 0.0;
    std::uint64_t g_last_color_frame_number = 0;
    bool g_has_depth = false;
    std::deque<imuData> g_pending_imu;

    rs2::align* g_align_to_color = nullptr;

    double nowSeconds()
    {
        using namespace std::chrono;
        return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
    }

    double frameTimestampMs(const rs2::frame& frame)
    {
        if (!frame) {
            return 0.0;
        }
        return static_cast<double>(frame.get_timestamp());
    }

    void resetState()
    {
        g_last_frame.release();
        g_last_depth_m.release();
        g_last_imu = imuData{};
        g_last_gyro_ts = 0.0;
        g_last_accel_ts = 0.0;
        g_last_emitted_imu_ts = -std::numeric_limits<double>::infinity();
        g_have_gyro = false;
        g_have_accel = false;
        g_prev_accel_sample = StampedVec3{};
        g_curr_accel_sample = StampedVec3{};
        g_pending_gyro.clear();
        g_last_color_ts_ms = 0.0;
        g_last_depth_ts_ms = 0.0;
        g_last_color_frame_number = 0;
        g_has_depth = false;
        g_pending_imu.clear();

        g_csv_has_header = false;
        g_csv_has_timestamp = false;
        g_csv_ts_col = -1;
        g_csv_gx_col = -1;
        g_csv_gy_col = -1;
        g_csv_gz_col = -1;
        g_csv_ax_col = -1;
        g_csv_ay_col = -1;
        g_csv_az_col = -1;
        g_csv_row_index = 0;
        g_csv_next_ts = 0.0;
    }

    double clamp(double x, double lo, double hi)
    {
        return std::max(lo, std::min(hi, x));
    }

    cv::Vec3d estimateAccelAt(double ts, bool allow_extrapolation)
    {
        if (!g_curr_accel_sample.valid) {
            return cv::Vec3d(0.0, 0.0, 0.0);
        }

        if (!g_prev_accel_sample.valid) {
            return g_curr_accel_sample.value;
        }

        const double dt = g_curr_accel_sample.ts - g_prev_accel_sample.ts;
        if (dt <= 1e-9 || !std::isfinite(dt)) {
            return g_curr_accel_sample.value;
        }

        double alpha = (ts - g_prev_accel_sample.ts) / dt;
        if (allow_extrapolation) {
            alpha = clamp(alpha, -0.25, 2.0);
        } else {
            alpha = clamp(alpha, 0.0, 1.0);
        }

        return g_prev_accel_sample.value * (1.0 - alpha) + g_curr_accel_sample.value * alpha;
    }

    bool emitImuSample(double ts, const cv::Vec3d& gyro, const cv::Vec3d& accel)
    {
        if (!(ts > g_last_emitted_imu_ts + 1e-9)) {
            return false;
        }

        imuData sample{};
        sample.ts = ts;
        sample.gyro = gyro;
        sample.accel = accel;

        g_last_imu = sample;
        g_pending_imu.push_back(sample);
        g_last_emitted_imu_ts = ts;
        return true;
    }

    bool flushPendingGyrosUpTo(double max_ts, bool allow_extrapolation)
    {
        if (!g_curr_accel_sample.valid) {
            return false;
        }

        bool emitted = false;
        while (!g_pending_gyro.empty() && g_pending_gyro.front().ts <= max_ts + 1e-9) {
            const PendingGyroSample gyro = g_pending_gyro.front();
            g_pending_gyro.pop_front();
            emitted |= emitImuSample(gyro.ts, gyro.gyro, estimateAccelAt(gyro.ts, allow_extrapolation));
        }
        return emitted;
    }

    std::string trim(const std::string& s)
    {
        const auto begin = std::find_if_not(s.begin(), s.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        });
        const auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }).base();
        if (begin >= end) {
            return std::string{};
        }
        return std::string(begin, end);
    }

    std::string normalizeKey(const std::string& s)
    {
        std::string out;
        out.reserve(s.size());
        for (unsigned char ch : s) {
            if (std::isalnum(ch) != 0) {
                out.push_back(static_cast<char>(std::tolower(ch)));
            }
        }
        return out;
    }

    bool tryParseDouble(const std::string& s, double* out)
    {
        if (out == nullptr) {
            return false;
        }

        const std::string t = trim(s);
        if (t.empty()) {
            return false;
        }

        try {
            size_t idx = 0;
            const double v = std::stod(t, &idx);
            if (idx != t.size()) {
                return false;
            }
            *out = v;
            return true;
        } catch (...) {
            return false;
        }
    }

    std::vector<std::string> splitCsvLine(const std::string& line)
    {
        std::vector<std::string> fields;
        std::string current;
        bool in_quotes = false;

        for (size_t i = 0; i < line.size(); ++i) {
            const char ch = line[i];

            if (ch == '"') {
                if (in_quotes && (i + 1) < line.size() && line[i + 1] == '"') {
                    current.push_back('"');
                    ++i;
                } else {
                    in_quotes = !in_quotes;
                }
                continue;
            }

            if (ch == ',' && !in_quotes) {
                fields.push_back(trim(current));
                current.clear();
                continue;
            }

            current.push_back(ch);
        }

        fields.push_back(trim(current));
        return fields;
    }

    bool lineLooksLikeHeader(const std::vector<std::string>& fields)
    {
        for (const auto& f : fields) {
            double dummy = 0.0;
            if (!tryParseDouble(f, &dummy)) {
                return true;
            }
        }
        return false;
    }

    bool isOneOf(const std::string& key, std::initializer_list<const char*> values)
    {
        for (const char* value : values) {
            if (key == value) {
                return true;
            }
        }
        return false;
    }

    bool configureCsvColumnsFromHeader(const std::vector<std::string>& fields)
    {
        g_csv_ts_col = -1;
        g_csv_gx_col = -1;
        g_csv_gy_col = -1;
        g_csv_gz_col = -1;
        g_csv_ax_col = -1;
        g_csv_ay_col = -1;
        g_csv_az_col = -1;

        for (int i = 0; i < static_cast<int>(fields.size()); ++i) {
            const std::string key = normalizeKey(fields[static_cast<size_t>(i)]);

            if (isOneOf(key, {"ts", "t", "time", "timestamp", "timesec", "timestampsec"})) {
                g_csv_ts_col = i;
            } else if (isOneOf(key, {"gx", "gyrox", "rawgx", "gyrorawx", "gyroraw", "gyroradxs"})) {
                g_csv_gx_col = i;
            } else if (isOneOf(key, {"gy", "gyroy", "rawgy", "gyrorawy", "gyroradys"})) {
                g_csv_gy_col = i;
            } else if (isOneOf(key, {"gz", "gyroz", "rawgz", "gyrorawz", "gyroradzs"})) {
                g_csv_gz_col = i;
            } else if (isOneOf(key, {"ax", "accx", "accelx", "rawax", "accrawx", "accelrawx"})) {
                g_csv_ax_col = i;
            } else if (isOneOf(key, {"ay", "accy", "accely", "raway", "accrawy", "accelrawy"})) {
                g_csv_ay_col = i;
            } else if (isOneOf(key, {"az", "accz", "accelz", "rawaz", "accrawz", "accelrawz"})) {
                g_csv_az_col = i;
            }
        }

        g_csv_has_timestamp = (g_csv_ts_col >= 0);
        return g_csv_gx_col >= 0 && g_csv_gy_col >= 0 && g_csv_gz_col >= 0 &&
               g_csv_ax_col >= 0 && g_csv_ay_col >= 0 && g_csv_az_col >= 0;
    }

    bool configureCsvColumnsWithoutHeader(const std::vector<std::string>& fields)
    {
        if (fields.size() >= 7) {
            g_csv_has_timestamp = true;
            g_csv_ts_col = 0;
            g_csv_gx_col = 1;
            g_csv_gy_col = 2;
            g_csv_gz_col = 3;
            g_csv_ax_col = 4;
            g_csv_ay_col = 5;
            g_csv_az_col = 6;
            return true;
        }

        if (fields.size() >= 6) {
            g_csv_has_timestamp = false;
            g_csv_ts_col = -1;
            g_csv_gx_col = 0;
            g_csv_gy_col = 1;
            g_csv_gz_col = 2;
            g_csv_ax_col = 3;
            g_csv_ay_col = 4;
            g_csv_az_col = 5;
            return true;
        }

        return false;
    }

    bool parseCsvImuFields(const std::vector<std::string>& fields, imuData* imu)
    {
        if (imu == nullptr) {
            return false;
        }

        auto readAt = [&](int idx, double* out) -> bool {
            if (idx < 0 || idx >= static_cast<int>(fields.size())) {
                return false;
            }
            return tryParseDouble(fields[static_cast<size_t>(idx)], out);
        };

        double gx = 0.0, gy = 0.0, gz = 0.0;
        double ax = 0.0, ay = 0.0, az = 0.0;
        if (!readAt(g_csv_gx_col, &gx) || !readAt(g_csv_gy_col, &gy) || !readAt(g_csv_gz_col, &gz) ||
            !readAt(g_csv_ax_col, &ax) || !readAt(g_csv_ay_col, &ay) || !readAt(g_csv_az_col, &az)) {
            return false;
        }

        double ts = 0.0;
        if (g_csv_has_timestamp) {
            if (!readAt(g_csv_ts_col, &ts)) {
                return false;
            }
        } else {
            ts = g_csv_next_ts;
            g_csv_next_ts += 1.0 / std::max(1e-9, g_config.imu.freq);
        }

        imu->ts = ts;
        imu->gyro = cv::Vec3d(gx, gy, gz);
        imu->accel = cv::Vec3d(ax, ay, az);
        return true;
    }

    bool readNextCsvRow(sourcePacket* packet)
    {
        if (!g_csv_started || !g_csv.is_open()) {
            return false;
        }

        std::string line;
        while (std::getline(g_csv, line)) {
            const std::string t = trim(line);
            if (t.empty() || t[0] == '#') {
                continue;
            }

            const std::vector<std::string> fields = splitCsvLine(t);
            if (fields.empty()) {
                continue;
            }

            if (g_csv_gx_col < 0) {
                g_csv_has_header = lineLooksLikeHeader(fields);
                const bool ok = g_csv_has_header
                    ? configureCsvColumnsFromHeader(fields)
                    : configureCsvColumnsWithoutHeader(fields);
                if (!ok) {
                    std::cerr << "CSV IMU invalido: no se pudieron identificar columnas ts/gx/gy/gz/ax/ay/az" << std::endl;
                    return false;
                }
                if (g_csv_has_header) {
                    continue;
                }
            }

            imuData imu{};
            if (!parseCsvImuFields(fields, &imu)) {
                continue;
            }

            ++g_csv_row_index;
            g_last_frame.release();
            g_last_depth_m.release();
            g_has_depth = false;
            g_last_imu = imu;
            g_last_color_ts_ms = 0.0;
            g_last_depth_ts_ms = 0.0;
            g_last_color_frame_number = g_csv_row_index;

            if (packet != nullptr) {
                packet->imu_data.clear();
                packet->imu_data.push_back(g_last_imu);
                packet->color.release();
                packet->depth.release();
                packet->new_color = false;
                packet->new_depth = false;
                packet->colorts_ms = 0.0;
                packet->depthts_ms = 0.0;
                packet->color_frame_number = g_last_color_frame_number;
            }
            return true;
        }

        return false;
    }

    bool updateFromColorFrame(const rs2::video_frame& color)
    {
        if (!color) {
            return false;
        }

        cv::Mat img(
            cv::Size(color.get_width(), color.get_height()),
            CV_8UC3,
            const_cast<void*>(color.get_data()),
            cv::Mat::AUTO_STEP
        );

        if (g_config.cam.rgb_format == 1) {
            cv::cvtColor(img, g_last_frame, cv::COLOR_RGB2BGR);
        } else {
            g_last_frame = img.clone();
        }

        g_last_color_ts_ms = frameTimestampMs(color);
        g_last_color_frame_number = color.get_frame_number();
        return true;
    }

    bool updateFromDepthFrame(const rs2::depth_frame& depth)
    {
        if (!depth || !g_need_depth) {
            return false;
        }

        cv::Mat depth_view(
            cv::Size(depth.get_width(), depth.get_height()),
            CV_16UC1,
            const_cast<void*>(depth.get_data()),
            cv::Mat::AUTO_STEP
        );

        const float scale = depth.get_units();
        depth_view.convertTo(g_last_depth_m, CV_32F, scale);
        g_last_depth_ts_ms = frameTimestampMs(depth);
        g_has_depth = !g_last_depth_m.empty();
        return g_has_depth;
    }

    bool updateFromMotionFrame(const rs2::motion_frame& motion, double max_emit_ts_sec)
    {
        if (!motion) {
            return false;
        }

        const rs2_vector data = motion.get_motion_data();
        const rs2_stream st = motion.get_profile().stream_type();
        const double ts = 1e-3 * frameTimestampMs(motion);

        if (st == RS2_STREAM_ACCEL) {
            g_prev_accel_sample = g_curr_accel_sample;
            g_curr_accel_sample.ts = ts;
            g_curr_accel_sample.value = cv::Vec3d(data.x, data.y, data.z);
            g_curr_accel_sample.valid = true;
            g_last_imu.accel = g_curr_accel_sample.value;
            g_last_accel_ts = ts;
            g_have_accel = true;
            return flushPendingGyrosUpTo(std::min(g_curr_accel_sample.ts, max_emit_ts_sec), false);
        }

        if (st != RS2_STREAM_GYRO) {
            return false;
        }

        g_last_imu.gyro = cv::Vec3d(data.x, data.y, data.z);
        g_last_gyro_ts = ts;
        g_have_gyro = true;

        if (!g_have_accel) {
            return false;
        }

        g_pending_gyro.push_back(PendingGyroSample{g_last_gyro_ts, g_last_imu.gyro});
        return flushPendingGyrosUpTo(std::min(g_curr_accel_sample.ts, max_emit_ts_sec), false);
    }

    bool unpackFrame(const rs2::frame& f, bool* updated_color, bool* updated_depth, bool* updated_imu)
    {
        bool updated = false;
        if (updated_color != nullptr) *updated_color = false;
        if (updated_depth != nullptr) *updated_depth = false;
        if (updated_imu != nullptr) *updated_imu = false;

        if (auto fs = f.as<rs2::frameset>()) {
            rs2::frameset aligned = fs;
            if (g_need_depth && g_align_to_color != nullptr) {
                aligned = g_align_to_color->process(fs);
            }

            if (auto color = aligned.get_color_frame()) {
                const bool ok = updateFromColorFrame(color);
                updated |= ok;
                if (updated_color != nullptr) *updated_color |= ok;
            }

            if (g_need_depth) {
                if (auto depth = aligned.get_depth_frame()) {
                    const bool ok = updateFromDepthFrame(depth);
                    updated |= ok;
                    if (updated_depth != nullptr) *updated_depth |= ok;
                }
            }

            const double max_emit_ts_sec =
                (updated_color != nullptr && *updated_color && g_last_color_ts_ms > 0.0)
                    ? (1e-3 * g_last_color_ts_ms)
                    : std::numeric_limits<double>::infinity();

            std::vector<rs2::motion_frame> motions;
            for (auto&& sub : fs) {
                if (auto motion = sub.as<rs2::motion_frame>()) {
                    motions.push_back(motion);
                }
            }
            std::sort(motions.begin(), motions.end(), [](const rs2::motion_frame& a, const rs2::motion_frame& b) {
                return frameTimestampMs(a) < frameTimestampMs(b);
            });

            for (const auto& motion : motions) {
                const bool ok = updateFromMotionFrame(motion, max_emit_ts_sec);
                updated |= ok;
                if (updated_imu != nullptr) *updated_imu |= ok;
            }

            return updated;
        }

        if (auto vf = f.as<rs2::video_frame>()) {
            if (vf.get_profile().stream_type() == RS2_STREAM_COLOR) {
                const bool ok = updateFromColorFrame(vf);
                updated |= ok;
                if (updated_color != nullptr) *updated_color |= ok;
            }
        }

        if (auto motion = f.as<rs2::motion_frame>()) {
            const bool ok = updateFromMotionFrame(motion, std::numeric_limits<double>::infinity());
            updated |= ok;
            if (updated_imu != nullptr) *updated_imu |= ok;
        }

        return updated;
    }

    void fillPacket(sourcePacket* packet,
                    bool updated_color,
                    bool updated_depth,
                    bool /*updated_imu*/)
    {
        if (updated_color && g_last_color_ts_ms > 0.0) {
            flushPendingGyrosUpTo(1e-3 * g_last_color_ts_ms, true);
        }

        if (packet == nullptr) {
            g_pending_imu.clear();
            return;
        }

        packet->color = g_last_frame.clone();
        packet->depth = g_last_depth_m.clone();
        packet->new_color = updated_color;
        packet->new_depth = updated_depth && g_need_depth && g_has_depth;
        packet->colorts_ms = g_last_color_ts_ms;
        packet->depthts_ms = g_last_depth_ts_ms;
        packet->color_frame_number = g_last_color_frame_number;
        packet->imu_data.clear();
        packet->imu_data.swap(g_pending_imu);
    }

    bool pollBagOnce(sourcePacket* packet)
    {
        if (!g_rs_started) {
            return false;
        }

        rs2::frameset frames;
        if (!g_pipe.try_wait_for_frames(&frames, 100)) {
            return false;
        }

        bool updated_color = false;
        bool updated_depth = false;
        bool updated_imu = false;
        frames.keep();
        const bool updated = unpackFrame(frames, &updated_color, &updated_depth, &updated_imu);

        fillPacket(packet, updated_color, updated_depth, updated_imu);
        return updated || !g_last_frame.empty() || g_last_imu.ts > 0.0;
    }

    bool pollRsCamOnce(sourcePacket* packet)
    {
        if (!g_rs_started) {
            return false;
        }

        rs2::frameset frames;
        if (!g_pipe.poll_for_frames(&frames)) {
            return false;
        }

        bool updated_color = false;
        bool updated_depth = false;
        bool updated_imu = false;
        const bool updated = unpackFrame(frames, &updated_color, &updated_depth, &updated_imu);

        fillPacket(packet, updated_color, updated_depth, updated_imu);
        return updated || !g_last_frame.empty() || g_last_imu.ts > 0.0;
    }
}

imuData port_imu;
HANDLE husb = INVALID_HANDLE_VALUE;
std::mutex mut;
std::thread readport;
std::atomic<bool> g_stop_readport{false};
std::atomic<bool> g_port_has_sample{false};
std::uint64_t g_port_sample_counter = 0;
std::uint64_t g_port_last_served_counter = 0;

void readPort()
{
    uint8_t buffer[256];
    std::string rx;
    double last_raw_ts = 0.0;
    double ts_scale = 0.0;

    while (!g_stop_readport.load()) {
        const int len = COMReceive(husb, 1, buffer);

        if (len <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        rx.append(reinterpret_cast<const char*>(buffer), static_cast<size_t>(len));
        if (rx.size() > 4096) {
            rx.erase(0, rx.size() - 1024);
        }

        size_t pos = 0;
        while ((pos = rx.find("\n")) != std::string::npos) {
            std::string line = rx.substr(0, pos);
            rx.erase(0, pos + 1);

            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }

            double ts_raw = 0.0;
            double gx = 0.0, gy = 0.0, gz = 0.0;
            double ax = 0.0, ay = 0.0, az = 0.0;
            const int parsed = std::sscanf(
                line.c_str(),
                "%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                &ts_raw, &gx, &gy, &gz, &ax, &ay, &az
            );

            if (parsed != 7) {
                Logger(WARN, "Parse failed: parsed=%d raw='%s'", parsed, line.c_str());
                continue;
            }

            if (ts_scale == 0.0 && last_raw_ts > 0.0) {
                const double d = ts_raw - last_raw_ts;
                if (d > 1000.0) {
                    ts_scale = 1e-6;
                } else if (d > 1.0) {
                    ts_scale = 1e-3;
                } else {
                    ts_scale = 1.0;
                }
            }
            if (ts_scale == 0.0) {
                ts_scale = 1.0;
            }
            last_raw_ts = ts_raw;

            std::lock_guard<std::mutex> lock(mut);
            port_imu.ts = ts_raw * ts_scale;
            port_imu.gyro = cv::Vec3d(gx, gy, gz);
            port_imu.accel = cv::Vec3d(ax, ay, az);
            ++g_port_sample_counter;
            g_port_has_sample.store(true);
        }
    }
}

bool initSourceManager(Config& config)
{
    closeSourceManager();

    g_config = config;
    g_initialized = false;
    g_need_depth = config.gen.calc_gt;

    try {
        resetState();

        switch (config.gen.in_type) {
            case FEED_BAG: {
                g_rs_cfg = rs2::config();
                g_rs_cfg.enable_device_from_file(config.gen.input, false);

                g_profile = g_pipe.start(g_rs_cfg);

                rs2::device dev = g_profile.get_device();
                rs2::playback pb = dev.as<rs2::playback>();
                pb.set_real_time(true);

                delete g_align_to_color;
                g_align_to_color = g_need_depth ? new rs2::align(RS2_STREAM_COLOR) : nullptr;

                g_rs_started = true;
                g_initialized = true;
                return true;
            }

            case FEED_RSCAM: {
                g_rs_cfg = rs2::config();

                const rs2_format color_format = (config.cam.rgb_format == 1) ? RS2_FORMAT_RGB8 : RS2_FORMAT_BGR8;
                g_rs_cfg.enable_stream(RS2_STREAM_COLOR,
                                       config.cam.w,
                                       config.cam.h,
                                       color_format,
                                       config.cam.fps);

                if (g_need_depth) {
                    g_rs_cfg.enable_stream(RS2_STREAM_DEPTH);
                }

                if (config.cam.has_imu) {
                    g_rs_cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
                    g_rs_cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
                }

                g_profile = g_pipe.start(g_rs_cfg);

                delete g_align_to_color;
                g_align_to_color = g_need_depth ? new rs2::align(RS2_STREAM_COLOR) : nullptr;

                g_rs_started = true;
                g_initialized = true;
                return true;
            }

            case FEED_RTSP: {
                g_cap.open(config.gen.input);
                if (!g_cap.isOpened()) {
                    std::cerr << "No se pudo abrir RTSP: " << config.gen.input << std::endl;
                    return false;
                }

                if (config.cam.w > 0) g_cap.set(cv::CAP_PROP_FRAME_WIDTH, config.cam.w);
                if (config.cam.h > 0) g_cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.cam.h);
                if (config.cam.fps > 0) g_cap.set(cv::CAP_PROP_FPS, config.cam.fps);

                g_cap_started = true;
                g_initialized = true;
                return true;
            }

            case FEED_MP4: {
                g_cap.open(config.gen.input);
                if (!g_cap.isOpened()) {
                    std::cerr << "No se pudo abrir MP4 file: " << config.gen.input << std::endl;
                    return false;
                }

                config.cam.w = static_cast<int>(g_cap.get(cv::CAP_PROP_FRAME_WIDTH));
                config.cam.h = static_cast<int>(g_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
                config.cam.fps = static_cast<int>(g_cap.get(cv::CAP_PROP_FPS));

                g_cap_started = true;
                g_initialized = true;
                return true;
            }

            case FEED_PCCAM: {
                g_cap.open(0);
                if (!g_cap.isOpened()) {
                    std::cerr << "No se pudo abrir PC camera" << std::endl;
                    return false;
                }

                if (config.cam.w > 0) g_cap.set(cv::CAP_PROP_FRAME_WIDTH, config.cam.w);
                if (config.cam.h > 0) g_cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.cam.h);
                if (config.cam.fps > 0) g_cap.set(cv::CAP_PROP_FPS, config.cam.fps);

                g_cap_started = true;
                g_initialized = true;
                return true;
            }

            case FEED_CSV: {
                g_csv.open(config.gen.input);
                if (!g_csv.is_open()) {
                    std::cerr << "No se pudo abrir CSV de entrada: " << config.gen.input << std::endl;
                    return false;
                }

                g_csv_started = true;
                g_initialized = true;
                return true;
            }

            case FEED_PORT: {
                std::wstring widestr(config.gen.input.begin(), config.gen.input.end());
                if (!COMInit(husb, widestr.c_str())) {
                    Logger(ERROR, "Error opening port %s", config.gen.input.c_str());
                    return false;
                }

                g_initialized = true;
                g_rs_started = false;
                g_cap_started = false;
                g_csv_started = false;

                {
                    std::lock_guard<std::mutex> lock(mut);
                    port_imu = imuData{};
                    g_port_sample_counter = 0;
                    g_port_last_served_counter = 0;
                }
                g_port_has_sample.store(false);
                g_stop_readport.store(false);
                readport = std::thread(readPort);
                return true;
            }

            default:
                std::cerr << "Tipo de feed no soportado" << std::endl;
                return false;
        }
    }
    catch (const rs2::error& e) {
        std::cerr << "RealSense error en initSourceManager: "
                  << e.get_failed_function() << "(" << e.get_failed_args() << "): " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Excepcion en initSourceManager: " << e.what() << std::endl;
        return false;
    }
}

void closeSourceManager()
{
    g_stop_readport.store(true);
    if (readport.joinable()) {
        try {
            readport.join();
        } catch (...) {
        }
    }

    if (husb != INVALID_HANDLE_VALUE) {
        try {
            COMDeInit(husb);
        } catch (...) {
        }
        husb = INVALID_HANDLE_VALUE;
    }
    g_port_has_sample.store(false);

    if (g_rs_started) {
        try {
            g_pipe.stop();
        } catch (...) {
        }
        g_rs_started = false;
    }

    if (g_cap_started) {
        g_cap.release();
        g_cap_started = false;
    }

    if (g_csv_started) {
        g_csv.close();
        g_csv_started = false;
    }

    if (g_align_to_color != nullptr) {
        delete g_align_to_color;
        g_align_to_color = nullptr;
    }

    resetState();
    g_initialized = false;
    g_need_depth = false;
}

bool getSourceManager(sourcePacket* packet)
{
    if (!g_initialized) {
        return false;
    }

    switch (g_config.gen.in_type) {
        case FEED_BAG: {
            if (pollBagOnce(packet)) {
                return true;
            }

            for (int i = 0; i < 20; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                if (pollBagOnce(packet)) {
                    return true;
                }
            }

            return false;
        }

        case FEED_RSCAM: {
            if (pollRsCamOnce(packet)) {
                return true;
            }

            for (int i = 0; i < 5; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (pollRsCamOnce(packet)) {
                    return true;
                }
            }

            return false;
        }

        case FEED_MP4:
        case FEED_RTSP:
        case FEED_PCCAM: {
            if (!g_cap_started) {
                return false;
            }

            cv::Mat img;
            if (!g_cap.read(img)) {
                return false;
            }

            g_last_frame = img.clone();
            g_last_depth_m.release();
            g_has_depth = false;

            g_last_imu = imuData{};
            g_last_imu.ts = nowSeconds();
            g_last_color_ts_ms = g_last_imu.ts * 1e3;
            g_last_depth_ts_ms = 0.0;
            ++g_last_color_frame_number;

            fillPacket(packet, true, false, false);
            return true;
        }

        case FEED_CSV:
            return readNextCsvRow(packet);

        case FEED_PORT: {
            if (packet != nullptr) {
                packet->imu_data.clear();
                packet->color.release();
                packet->depth.release();
                packet->new_color = false;
                packet->new_depth = false;
                packet->colorts_ms = 0.0;
                packet->depthts_ms = 0.0;
                packet->color_frame_number = 0;
            }

            if (!g_port_has_sample.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                return true;
            }

            imuData local_imu;
            bool has_new = false;
            std::uint64_t local_counter = 0;
            {
                std::lock_guard<std::mutex> lock(mut);
                local_imu = port_imu;
                local_counter = g_port_sample_counter;
                if (g_port_sample_counter != g_port_last_served_counter) {
                    has_new = true;
                    g_port_last_served_counter = g_port_sample_counter;
                }
            }

            if (packet != nullptr && has_new && local_imu.ts > 0.0) {
                packet->imu_data.push_back(local_imu);
                packet->color_frame_number = local_counter;
            }

            if (!has_new) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return true;
        }

        default:
            return false;
    }
}

bool getSourceManager(cv::Mat* frame, imuData* imu)
{
    sourcePacket packet;
    const bool ok = getSourceManager(&packet);
    if (!ok) {
        return false;
    }

    if (frame != nullptr) {
        *frame = packet.color.clone();
    }

    if (imu != nullptr) {
        *imu = packet.imu_data.empty() ? imuData{} : packet.imu_data.back();
    }

    return true;
}
