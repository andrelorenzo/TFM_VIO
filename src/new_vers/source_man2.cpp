// #include "source_man.hpp"
#include "config.hpp"
#include "csv_logger.hpp"

// seconds
#include "seconds/comms_common.h"

// thirds
#include "librealsense2/rs.hpp"

#include <thread>
#include <deque>
#include <vector>
#include <cmath>
#include <utility>
#include <mutex>

#define MAX_HIST 5000

static rs2::pipeline pipe;
static rs2::frame_queue queue(MAX_HIST * 2);
static std::thread poll_thread;
static HANDLE husb = INVALID_HANDLE_VALUE;

static std::mutex dout_mutex;
static std::deque<SourceIn> vout;

static Config source_config;

static void updateSource2(Config * config);

static double normTs(double raw_ts) {
    static bool init = false;
    static double first_raw_ts = 0.0;
    static double last_raw_ts = 0.0;
    static double wrap_offset = 0.0;

    const double WRAP_THRESHOLD_MS = 1000.0;

    if (!init) {
        init = true;
        first_raw_ts = raw_ts;
        last_raw_ts = raw_ts;
        wrap_offset = 0.0;
        return 0.0;
    }

    if (raw_ts + WRAP_THRESHOLD_MS < last_raw_ts) wrap_offset += last_raw_ts;
    last_raw_ts = raw_ts;

    return wrap_offset + raw_ts - first_raw_ts;
}

static bool interpolateImu(const std::deque<std::pair<double, vec3>>& buf, double t, vec3& out) {
    if (buf.size() < 2) return false;
    if (t < buf.front().first || t > buf.back().first) return false;

    for (size_t i = 0; i + 1 < buf.size(); ++i) {
        const double ta = buf[i].first;
        const double tb = buf[i + 1].first;
        const vec3& va = buf[i].second;
        const vec3& vb = buf[i + 1].second;

        if (ta <= t && t <= tb) {
            const double dt = tb - ta;
            if (std::abs(dt) < 1e-9) {
                out = va;
                return true;
            }

            const double a = (t - ta) / dt;
            out = vec3(va.x() + (vb.x() - va.x()) * a, va.y() + (vb.y() - va.y()) * a, va.z() + (vb.z() - va.z()) * a);
            return true;
        }
    }

    return false;
}

static void trimImuBuffer(std::deque<std::pair<double, vec3>>& buf, double t) {
    while (buf.size() >= 3 && buf[1].first < t) buf.pop_front();
}

bool initSource2(Config config) {
    source_config = config;

    rs2::config bag_config;

    if (source_config.gen.type == SOURCE_BAG) {
        bag_config.enable_device_from_file(source_config.gen.input, false);
        bag_config.enable_stream(RS2_STREAM_ACCEL);
        bag_config.enable_stream(RS2_STREAM_GYRO);

        if (source_config.gen.color_on) {
            bag_config.enable_stream(RS2_STREAM_COLOR);
            Logger(INFO, "Color feed enabled");
        }

        rs2::pipeline_profile profile = pipe.start(bag_config, [&](const rs2::frame& f) { queue.enqueue(f); });

        for (const rs2::stream_profile& sp : profile.get_streams()) {
            const rs2_stream st = sp.stream_type();

            if (st == RS2_STREAM_COLOR) {
                rs2::video_stream_profile vsp = sp.as<rs2::video_stream_profile>();

                if (vsp) {
                    source_config.cam.w = static_cast<uint32_t>(vsp.width());
                    source_config.cam.h = static_cast<uint32_t>(vsp.height());

                    if (source_config.cam.fps > static_cast<double>(vsp.fps())) source_config.cam.fps = static_cast<double>(vsp.fps());

                    const rs2_format fmt = vsp.format();
                    if (fmt == RS2_FORMAT_RGB8 || fmt == RS2_FORMAT_RGBA8) source_config.cam.is_rgb = true;
                    else if (fmt == RS2_FORMAT_BGR8 || fmt == RS2_FORMAT_BGRA8) source_config.cam.is_rgb = false;
                }
            }
        }
    }
    else if (source_config.gen.type == SOURCE_PORT) {
        std::wstring widestr(source_config.gen.input.begin(), source_config.gen.input.end());

        if (!COMInit(husb, widestr.c_str())) {
            Logger(ERROR, "initSourceMan => Source %s could not be opened", source_config.gen.input.c_str());
            return false;
        }
    }
    else {
        Logger(ERROR, "Unreachable: Source not available on init");
        return false;
    }

    poll_thread = std::thread(updateSource2, &source_config);
    return true;
}

static void pollRealSense(double * init_ts) {
    static std::deque<std::pair<double, vec3>> acc_buf;
    static std::deque<std::pair<double, vec3>> gyr_buf;
    static double last_frame_ts = -1.0;

    const size_t MAX_IMU_BUF = 500;
    const double EPS_MS = 1e-6;

    rs2::frame fframe;

    while (queue.poll_for_frame(&fframe)) {
        rs2::frameset fs = fframe.as<rs2::frameset>();

        if (!fs) {
            rs2::motion_frame mf = fframe.as<rs2::motion_frame>();
            if (!mf) continue;

            rs2_stream stream_type = mf.get_profile().stream_type();
            double ts = normTs(mf.get_timestamp());
            rs2_vector v = mf.get_motion_data();
            vec3 value(v.x, v.y, v.z);

            switch (stream_type) {
                case RS2_STREAM_ACCEL: {
                    acc_buf.emplace_back(ts, value);
                    while (acc_buf.size() > MAX_IMU_BUF) acc_buf.pop_front();
                } break;

                case RS2_STREAM_GYRO: {
                    gyr_buf.emplace_back(ts, value);
                    while (gyr_buf.size() > MAX_IMU_BUF) gyr_buf.pop_front();
                } break;

                default: break;
            }

            continue;
        }

        rs2::video_frame vf = fs.get_color_frame();
        if (!vf) continue;

        double frame_ts = normTs(vf.get_timestamp());

        if (last_frame_ts < 0.0) {
            last_frame_ts = frame_ts;
            continue;
        }

        if (source_config.cam.fps > 0.0) {
            const double min_frame_dt = 1000.0 / source_config.cam.fps;
            if ((frame_ts - last_frame_ts) + EPS_MS < min_frame_dt) continue;
        }

        if (acc_buf.size() < 2 || gyr_buf.size() < 2) continue;
        if (frame_ts > acc_buf.back().first || frame_ts > gyr_buf.back().first) continue;

        std::vector<double> target_ts;

        for (const auto& g : gyr_buf) {
            if (g.first > last_frame_ts + EPS_MS && g.first < frame_ts - EPS_MS) target_ts.emplace_back(g.first);
        }

        if (target_ts.empty() || std::abs(target_ts.back() - frame_ts) > EPS_MS) target_ts.emplace_back(frame_ts);

        SourceIn packet;
        packet.frame_dtms = frame_ts - last_frame_ts;
        packet.frame_tsms = frame_ts;

        double prev_t = last_frame_ts;

        for (double t : target_ts) {
            vec3 acc_i;
            vec3 gyr_i;

            bool ok_acc = interpolateImu(acc_buf, t, acc_i);
            bool ok_gyr = interpolateImu(gyr_buf, t, gyr_i);

            if (!ok_acc || !ok_gyr) continue;
            if (t <= prev_t + EPS_MS) continue;

            ImuSample s;
            s.ts = t;
            s.dt = t - prev_t;
            s.vgyr = gyr_i;
            s.vacc = acc_i;

            packet.imu.emplace_back(s);
            prev_t = t;
        }

        if (packet.imu.empty()) continue;

        double imu_dt_sum = 0.0;
        for (const ImuSample& s : packet.imu) imu_dt_sum += s.dt;

        if (std::abs(imu_dt_sum - packet.frame_dtms) > 1e-3) {
            Logger(WARN, "IMU block dt mismatch: sum=%.6f frame_dt=%.6f t0=%.6f t1=%.6f prev_frame=%.6f frame=%.6f n=%zu", imu_dt_sum, packet.frame_dtms, packet.imu.front().ts, packet.imu.back().ts, last_frame_ts, frame_ts, packet.imu.size());
        }

        cv::Mat out;
        cv::Mat img_rgb(cv::Size(vf.get_width(), vf.get_height()), CV_8UC3, (void*)vf.get_data(), cv::Mat::AUTO_STEP);

        if (source_config.cam.is_rgb) cv::cvtColor(img_rgb, out, cv::COLOR_RGB2BGR);
        else out = img_rgb;

        packet.frame = out.clone();

        trimImuBuffer(acc_buf, frame_ts);
        trimImuBuffer(gyr_buf, frame_ts);

        {
            std::lock_guard<std::mutex> lock(dout_mutex);
            vout.emplace_back(packet);
            while (vout.size() > MAX_HIST) vout.pop_front();
        }

        last_frame_ts = frame_ts;
    }
}

static void updateSource2(Config * config) {
    static double init_ts = 0.0;

    while (true) {
        if (config->gen.type == SOURCE_BAG) pollRealSense(&init_ts);
        else if (config->gen.type == SOURCE_PORT) {}
        else {
            Logger(ERROR, "Unreachable: Source not available while polling");
            break;
        }
    }
}

int getSource2(SourceIn * out) {
    if (!out) return -1;

    if (source_config.gen.type == SOURCE_BAG) {
        std::lock_guard<std::mutex> lock(dout_mutex);
        if (vout.empty()) return 0;
        *out = vout.front();
        vout.pop_front();
        return 1;
    }

    return 0;
}
