#include "source_man.hpp"
#include "csv_logger.hpp"

// seconds
#include "seconds/comms_common.h"

// thirds
#include "librealsense2/rs.hpp"

// std
#include <vector>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <atomic>

#define MAX_HIST 5000
#define MAX_FRAME_QUEUE 64

static std::atomic<SourceManType> source_state{SourceManType::SOURCEMAN_OK};
static std::atomic<bool> run_polling{false};
static bool g_source_man_init = false;
static std::mutex mut;
static std::condition_variable g_cv;
static SourceIn gout;
static std::deque<SourceIn> g_frame_queue;
static std::thread pollThread;
static bool g_bundle_imu_with_frame = false;
static bool g_imu_only_mode = false;
static double g_imu_batch_ms = 0.0;
static size_t g_frame_packets_dropped = 0;

static cv::VideoCapture cap;
static HANDLE husb = INVALID_HANDLE_VALUE;
static SOCKET hudp = INVALID_SOCKET;
static double old_frame_ts = -1.0;
static rs2::frame_queue queue(MAX_HIST*2);
static rs2::pipeline pipe;
static rs2::align* g_align_to_color = nullptr;
static CSVLogger reader;
static std::vector<std::string> rheader;
static double rs_ts_origin_ms = -1.0;
static double rs_ts_wrap_ms = 0.0;
static double rs_last_raw_ms[4] = {-1.0, -1.0, -1.0, -1.0};
static double rs_last_norm_ms[4] = {-1.0, -1.0, -1.0, -1.0};
static bool g_rotate_imu_to_color = false;
static mat3 g_R_ci = mat3::Identity();

enum RsTsSlot { RS_TS_COLOR = 0, RS_TS_DEPTH = 1, RS_TS_ACCEL = 2, RS_TS_GYRO = 3 };

static void fillConfigFromProfile(const rs2::pipeline_profile& profile, Config* config);
static void pushImuSampleLocked(const double* recv);
static void receiveUdpImuLocked();

static vec3 rotateImuToColorFrame(const vec3& v) {
    return g_rotate_imu_to_color ? (g_R_ci * v) : v;
}

static void resetRsTimestamps() {
    rs_ts_origin_ms = -1.0;
    rs_ts_wrap_ms = 0.0;
    for (int i = 0; i < 4; ++i) {
        rs_last_raw_ms[i] = -1.0;
        rs_last_norm_ms[i] = -1.0;
    }
}

static double normalizeRsTsMs(const double raw_ms, const RsTsSlot slot) {
    if (rs_ts_origin_ms < 0.0) rs_ts_origin_ms = raw_ms;
    if (rs_last_raw_ms[slot] >= 0.0 && raw_ms + 100.0 < rs_last_raw_ms[slot]) {
        rs_ts_wrap_ms += (rs_last_norm_ms[slot] - (raw_ms + rs_ts_wrap_ms - rs_ts_origin_ms)) + 1.0;
    }

    double norm_ms = raw_ms + rs_ts_wrap_ms - rs_ts_origin_ms;
    if (rs_last_norm_ms[slot] >= 0.0 && norm_ms < rs_last_norm_ms[slot]) {
        norm_ms = rs_last_norm_ms[slot];
    }

    rs_last_raw_ms[slot] = raw_ms;
    rs_last_norm_ms[slot] = norm_ms;
    return norm_ms;
}

static void clearSourceOutLocked(SourceIn* data) {
    if (data == nullptr) return;

    data->frame.release();
    data->frame_tsms = 0.0;
    data->frame_dtms = 0.0;
    data->depth.release();
    data->depth_tsms = 0.0;

    data->acc.clear();
    data->acc_tsms.clear();
    data->acc_dtms.clear();
    data->gyr.clear();
    data->gyr_tsms.clear();
    data->gyr_dtms.clear();
}

static size_t countSamplesUpToTsLocked(const std::deque<double>& tsms, double cut_tsms) {
    size_t count = 0;
    while (count < tsms.size() && tsms[count] <= cut_tsms) {
        ++count;
    }
    return count;
}

template <typename T>
static void moveDequePrefixLocked(std::deque<T>& src, std::deque<T>& dst, size_t count) {
    for (size_t i = 0; i < count && !src.empty(); ++i) {
        dst.emplace_back(src.front());
        src.pop_front();
    }
}

static void rebuildDtmsLocked(const std::deque<double>& tsms, std::deque<double>& dtms) {
    dtms.clear();
    for (size_t i = 1; i < tsms.size(); ++i) {
        dtms.emplace_back(tsms[i] - tsms[i - 1]);
    }
}

static void appendSyntheticFrameSampleLocked(std::deque<vec3>& out_data,
                                             std::deque<double>& out_tsms,
                                             const std::deque<vec3>& gout_data,
                                             const std::deque<double>& gout_tsms,
                                             const double frame_tsms) {
    if (out_data.empty() || out_tsms.empty()) return;
    if (std::abs(out_tsms.back() - frame_tsms) < 1e-9) return;

    // If we already have two samples before the frame, let pre_int close the
    // interval itself. We only synthesize here for the problematic case where
    // the frame falls between the last emitted sample and the first sample that
    // still remains queued in gout.
    if (out_data.size() >= 2 && out_tsms.back() < frame_tsms) return;
    if (gout_data.empty() || gout_tsms.empty()) return;

    const double t0 = out_tsms.back();
    const double t1 = gout_tsms.front();
    if (!(t0 < frame_tsms && frame_tsms <= t1)) return;
    if (t1 <= t0) return;

    const double alpha = (frame_tsms - t0) / (t1 - t0);
    out_data.emplace_back((1.0 - alpha) * out_data.back() + alpha * gout_data.front());
    out_tsms.emplace_back(frame_tsms);
}

static bool frameQueueSaturatedLocked() {
    return g_frame_queue.size() >= MAX_FRAME_QUEUE;
}

static bool imuBatchReadyLocked() {
    if (gout.acc.size() < 2 || gout.gyr.size() < 2) return false;
    if (g_imu_batch_ms <= 0.0) return true;
    const double acc_span_ms = gout.acc_tsms.back() - gout.acc_tsms.front();
    const double gyr_span_ms = gout.gyr_tsms.back() - gout.gyr_tsms.front();
    return acc_span_ms >= g_imu_batch_ms && gyr_span_ms >= g_imu_batch_ms;
}

static void moveAccumulatorToPacketLocked(SourceIn* pkt) {
    if (pkt == nullptr) return;

    // Each queued SourceIn owns one frame plus all IMU collected before it.
    // Keep the last sample as overlap in the accumulator so the next packet
    // still starts from the previous boundary.
    pkt->acc = gout.acc;
    pkt->acc_tsms = gout.acc_tsms;
    pkt->acc_dtms = gout.acc_dtms;
    pkt->gyr = gout.gyr;
    pkt->gyr_tsms = gout.gyr_tsms;
    pkt->gyr_dtms = gout.gyr_dtms;

    std::deque<vec3> next_acc;
    std::deque<double> next_acc_tsms;
    std::deque<vec3> next_gyr;
    std::deque<double> next_gyr_tsms;

    if (!gout.acc.empty() && !gout.acc_tsms.empty()) {
        next_acc.emplace_back(gout.acc.back());
        next_acc_tsms.emplace_back(gout.acc_tsms.back());
    }
    if (!gout.gyr.empty() && !gout.gyr_tsms.empty()) {
        next_gyr.emplace_back(gout.gyr.back());
        next_gyr_tsms.emplace_back(gout.gyr_tsms.back());
    }

    gout.acc.swap(next_acc);
    gout.acc_tsms.swap(next_acc_tsms);
    gout.acc_dtms.clear();
    gout.gyr.swap(next_gyr);
    gout.gyr_tsms.swap(next_gyr_tsms);
    gout.gyr_dtms.clear();
}

static void pushFramePacketLocked(std::unique_lock<std::mutex>& lock, SourceIn&& pkt) {
    if (!run_polling.load(std::memory_order_relaxed)) return;
    if (frameQueueSaturatedLocked()) {
        // For visual debugging we prefer dropping the oldest pending frame over
        // stalling the producer forever when the consumer cannot keep up.
        g_frame_queue.pop_front();
        ++g_frame_packets_dropped;
        if ((g_frame_packets_dropped % 25U) == 1U) {
            Logger(WARN, "SourceMan dropping old frame packets due to backlog (dropped=%zu, q=%zu)",
                g_frame_packets_dropped, g_frame_queue.size());
        }
    }
    g_frame_queue.emplace_back(std::move(pkt));
    g_cv.notify_all();
}

template <typename T>
static void retainLastSampleAsOverlapLocked(const std::deque<T>& out_data,
                                            const std::deque<double>& out_tsms,
                                            std::deque<T>& gout_data,
                                            std::deque<double>& gout_tsms,
                                            std::deque<double>& gout_dtms) {
    if (out_data.empty() || out_tsms.empty()) return;
    if (!gout_tsms.empty() && std::abs(gout_tsms.front() - out_tsms.back()) < 1e-9) return;

    gout_data.push_front(out_data.back());
    gout_tsms.push_front(out_tsms.back());
    rebuildDtmsLocked(gout_tsms, gout_dtms);
}

static void pushImuSampleLocked(const double* recv) {
    if (!gout.gyr_tsms.empty()) {
        gout.gyr_dtms.emplace_back(recv[0] - gout.gyr_tsms.back());
        if (gout.gyr_dtms.size() > MAX_HIST) gout.gyr_dtms.pop_front();
    }
    gout.gyr_tsms.emplace_back(recv[0]);
    if (gout.gyr_tsms.size() > MAX_HIST) gout.gyr_tsms.pop_front();

    gout.gyr.emplace_back(vec3(recv[1], recv[2], recv[3]));
    if (gout.gyr.size() > MAX_HIST) gout.gyr.pop_front();

    if (!gout.acc_tsms.empty()) {
        gout.acc_dtms.emplace_back(recv[0] - gout.acc_tsms.back());
        if (gout.acc_dtms.size() > MAX_HIST) gout.acc_dtms.pop_front();
    }
    gout.acc_tsms.emplace_back(recv[0]);
    if (gout.acc_tsms.size() > MAX_HIST) gout.acc_tsms.pop_front();

    gout.acc.emplace_back(vec3(recv[4], recv[5], recv[6]));
    if (gout.acc.size() > MAX_HIST) gout.acc.pop_front();
    g_cv.notify_all();
}

static void receiveUdpImuLocked() {

    if (hudp == INVALID_SOCKET) {
        return;
    }

    uint8_t buff[2048];
    while (true) {
        const int len = UDPReceive(hudp, buff, sizeof(buff) - 1);
        if (len <= 0) {
            break;
        }

        char* data = reinterpret_cast<char*>(buff);
        data[len] = '\0';
        char* line = data;

        for (int i = 0; i < len; ++i) {
            if (data[i] != '\n') {
                continue;
            }

            data[i] = '\0';
            if (i > 0 && data[i - 1] == '\r') {
                data[i - 1] = '\0';
            }

            if (*line != '\0') {
                double recv[7] = {0.0};
                const int parsed = std::sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                    &recv[0], &recv[1], &recv[2], &recv[3], &recv[4], &recv[5], &recv[6]);
                if (parsed == 7) {
                    pushImuSampleLocked(recv);
                }
            }

            line = data + i + 1;
        }

        if (*line != '\0') {
            const size_t line_len = std::strlen(line);
            if (line_len > 0 && line[line_len - 1] == '\r') {
                line[line_len - 1] = '\0';
            }

            double recv[7] = {0.0};
            const int parsed = std::sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                &recv[0], &recv[1], &recv[2], &recv[3], &recv[4], &recv[5], &recv[6]);
            if (parsed == 7) {
                pushImuSampleLocked(recv);
            }
        }
    }
}

static void pollingThread(Config * config){
    std::string port_buffer;
    double csv_prev_ts = -1.0;
    auto mp4_last_tick = std::chrono::steady_clock::now();

    while(run_polling.load(std::memory_order_relaxed)){
        switch(config->gen.type){
            // BUG[170426-214851](60):Creo que se reproduce en bucle, hay una opcion para desactivar eso
            case SOURCE_RSCAM:
            case SOURCE_BAG:{
                rs2::frame fframe;
                bool any_frame = false;

                while (queue.poll_for_frame(&fframe)) {
                    any_frame = true;
                    if (rs2::frameset fs = fframe.as<rs2::frameset>()) {
                        rs2::frameset aligned = fs;
                        if (g_align_to_color != nullptr) {
                            aligned = g_align_to_color->process(fs);
                        }

                        bool keep_color = false;
                        SourceIn frame_pkt;

                        if (config->gen.color_on) {
                            rs2::video_frame vf = aligned.get_color_frame();
                            if (!vf) continue;
                            const double frame_ts_ms = normalizeRsTsMs(vf.get_timestamp(), RS_TS_COLOR);
                            if (config->cam.fps <= 0.0 || old_frame_ts < 0.0 || (frame_ts_ms - old_frame_ts) >= (1000.0 / config->cam.fps)) {
                                const rs2_format fmt = vf.get_profile().format();
                                cv::Mat out;

                                if (fmt == RS2_FORMAT_RGB8) {
                                    cv::Mat img_rgb(
                                        cv::Size(vf.get_width(), vf.get_height()),
                                        CV_8UC3,
                                        (void*)vf.get_data(),
                                        cv::Mat::AUTO_STEP
                                    );
                                    cv::cvtColor(img_rgb, out, cv::COLOR_RGB2BGR);
                                }
                                else if (fmt == RS2_FORMAT_BGR8) {
                                    cv::Mat img_bgr(
                                        cv::Size(vf.get_width(), vf.get_height()),
                                        CV_8UC3,
                                        (void*)vf.get_data(),
                                        cv::Mat::AUTO_STEP
                                    );
                                    out = img_bgr.clone();
                                }
                                else {
                                    out.release();
                                }

                                if (!out.empty()) {
                                    frame_pkt.frame = out.clone();
                                    frame_pkt.frame_dtms = (old_frame_ts < 0.0) ? 0.0 : (frame_ts_ms - old_frame_ts);
                                    frame_pkt.frame_tsms = frame_ts_ms;
                                    old_frame_ts = frame_ts_ms;
                                    keep_color = true;
                                }
                            }
                        }

                        if (keep_color) {
                            if (rs2::depth_frame df = aligned.get_depth_frame()) {
                                cv::Mat depth_view(
                                    cv::Size(df.get_width(), df.get_height()),
                                    CV_16UC1,
                                    const_cast<void*>(df.get_data()),
                                    cv::Mat::AUTO_STEP
                                );

                                const float scale = df.get_units();
                                cv::Mat out;
                                depth_view.convertTo(out, CV_32F, scale);

                                frame_pkt.depth = out.clone();
                                frame_pkt.depth_tsms = normalizeRsTsMs(df.get_timestamp(), RS_TS_DEPTH);
                            } else {
                                frame_pkt.depth.release();
                                frame_pkt.depth_tsms = 0.0;
                            }
                        }

                        for (const rs2::frame& subf : fs) {
                            rs2_stream stream_type = subf.get_profile().stream_type();

                            switch (stream_type) {
                                case RS2_STREAM_ACCEL: {
                                    rs2::motion_frame mf = subf.as<rs2::motion_frame>();
                                    if (!mf) continue;
                                    rs2_vector v = mf.get_motion_data();
                                    std::lock_guard<std::mutex> lock(mut);
                                    gout.acc.emplace_back(rotateImuToColorFrame(vec3(v.x, v.y, v.z)));
                                    if (gout.acc.size() > MAX_HIST) gout.acc.pop_front();

                                    const double ts_ms = normalizeRsTsMs(mf.get_timestamp(), RS_TS_ACCEL);
                                    if(gout.acc_tsms.size() > 0)gout.acc_dtms.emplace_back(ts_ms-gout.acc_tsms[gout.acc_tsms.size()-1]);
                                    if(gout.acc_dtms.size() > MAX_HIST) gout.acc_dtms.pop_front();

                                    gout.acc_tsms.emplace_back(ts_ms);
                                    if(gout.acc_tsms.size() > MAX_HIST) gout.acc_tsms.pop_front();
                                    
                                    source_state.store(SOURCEMAN_OK, std::memory_order_relaxed);
                                    g_cv.notify_all();
                                } break;

                                case RS2_STREAM_GYRO: {
                                    rs2::motion_frame mf = subf.as<rs2::motion_frame>();
                                    if (!mf) continue;
                                    rs2_vector v = mf.get_motion_data();
                                    std::lock_guard<std::mutex> lock(mut);
                                    gout.gyr.emplace_back(rotateImuToColorFrame(vec3(v.x, v.y, v.z)));
                                    if (gout.gyr.size() > MAX_HIST) gout.gyr.pop_front();

                                    const double ts_ms = normalizeRsTsMs(mf.get_timestamp(), RS_TS_GYRO);
                                    if(gout.gyr_tsms.size() > 0)gout.gyr_dtms.emplace_back(ts_ms-gout.gyr_tsms[gout.gyr_tsms.size()-1]);
                                    if(gout.gyr_dtms.size() > MAX_HIST) gout.gyr_dtms.pop_front();

                                    gout.gyr_tsms.emplace_back(ts_ms);
                                    if (gout.gyr_tsms.size() > MAX_HIST) gout.gyr_tsms.pop_front();
                                    source_state.store(SOURCEMAN_OK, std::memory_order_relaxed);
                                    g_cv.notify_all();
                                } break;

                                case RS2_STREAM_COLOR:
                                case RS2_STREAM_DEPTH:
                                    break;

                                default:
                                    break;
                            }
                        }

                        if (keep_color) {
                            std::unique_lock<std::mutex> lock(mut);
                            moveAccumulatorToPacketLocked(&frame_pkt);
                            pushFramePacketLocked(lock, std::move(frame_pkt));
                            source_state.store(SOURCEMAN_OK, std::memory_order_relaxed);
                        }
                    }
                    else {
                        rs2_stream stream_type = fframe.get_profile().stream_type();

                        switch (stream_type) {
                            case RS2_STREAM_ACCEL: {
                                rs2::motion_frame mf = fframe.as<rs2::motion_frame>();
                                if (!mf) continue;
                                rs2_vector v = mf.get_motion_data();
                                // Logger(DEBUG, "Getting data from ACCEL");

                                std::lock_guard<std::mutex> lock(mut);
                                gout.acc.emplace_back(rotateImuToColorFrame(vec3(v.x, v.y, v.z)));
                                if (gout.acc.size() > MAX_HIST) gout.acc.pop_front();

                                const double ts_ms = normalizeRsTsMs(mf.get_timestamp(), RS_TS_ACCEL);
                                if(gout.acc_tsms.size() > 0)gout.acc_dtms.emplace_back(ts_ms-gout.acc_tsms[gout.acc_tsms.size()-1]);
                                if(gout.acc_dtms.size() > MAX_HIST) gout.acc_dtms.pop_front();

                                gout.acc_tsms.emplace_back(ts_ms);
                                if (gout.acc_tsms.size() > MAX_HIST) gout.acc_tsms.pop_front();
                                source_state.store(SOURCEMAN_OK, std::memory_order_relaxed);
                                g_cv.notify_all();
                            } break;

                            case RS2_STREAM_GYRO: {
                                rs2::motion_frame mf = fframe.as<rs2::motion_frame>();
                                if (!mf) continue;
                                rs2_vector v = mf.get_motion_data();
                                // Logger(DEBUG, "Getting data from GYRO");

                                std::lock_guard<std::mutex> lock(mut);
                                gout.gyr.emplace_back(rotateImuToColorFrame(vec3(v.x, v.y, v.z)));
                                if (gout.gyr.size() > MAX_HIST) gout.gyr.pop_front();

                                const double ts_ms = normalizeRsTsMs(mf.get_timestamp(), RS_TS_GYRO);
                                if(gout.gyr_tsms.size() > 0)gout.gyr_dtms.emplace_back(ts_ms-gout.gyr_tsms[gout.gyr_tsms.size()-1]);
                                if(gout.gyr_dtms.size() > MAX_HIST) gout.gyr_dtms.pop_front();

                                gout.gyr_tsms.emplace_back(ts_ms);
                                if (gout.gyr_tsms.size() > MAX_HIST) gout.gyr_tsms.pop_front();
                                source_state.store(SOURCEMAN_OK, std::memory_order_relaxed);
                                g_cv.notify_all();
                            } break;

                            default:
                                break;
                        }
                    }
                }

                if (!any_frame) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } break;

            case SOURCE_CSV:{
                std::vector<double> row;
                if(!reader.readRow(&row)){
                    source_state.store(SOURCEMAN_EOF, std::memory_order_relaxed);
                    g_cv.notify_all();
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                if(row.size() < 7){
                    Logger(WARN, "CSV row too small: %zu", row.size());
                    continue;
                }

                if(csv_prev_ts >= 0.0){
                    const double dt_ms = row[0] - csv_prev_ts;
                    if(dt_ms > 0.0 && dt_ms < 1000.0){
                        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(dt_ms)));
                    }
                }
                csv_prev_ts = row[0];

                std::lock_guard<std::mutex> lock(mut);

                gout.frame.release();
                gout.frame_tsms = 0.0;
                gout.frame_dtms = 0.0;
                gout.depth.release();
                gout.depth_tsms = 0.0;

                if (!gout.acc_tsms.empty()) {
                    gout.acc_dtms.emplace_back(row[0] - gout.acc_tsms.back());
                    if (gout.acc_dtms.size() > MAX_HIST) gout.acc_dtms.pop_front();
                }
                gout.acc_tsms.emplace_back(row[0]);
                if (gout.acc_tsms.size() > MAX_HIST) gout.acc_tsms.pop_front();

                if (!gout.gyr_tsms.empty()) {
                    gout.gyr_dtms.emplace_back(row[0] - gout.gyr_tsms.back());
                    if (gout.gyr_dtms.size() > MAX_HIST) gout.gyr_dtms.pop_front();
                }
                gout.gyr_tsms.emplace_back(row[0]);
                if (gout.gyr_tsms.size() > MAX_HIST) gout.gyr_tsms.pop_front();

                gout.gyr.emplace_back(vec3(row[1], row[2], row[3]));
                if (gout.gyr.size() > MAX_HIST) gout.gyr.pop_front();

                gout.acc.emplace_back(vec3(row[4], row[5], row[6]));
                if (gout.acc.size() > MAX_HIST) gout.acc.pop_front();

                source_state.store(SOURCEMAN_OK, std::memory_order_relaxed);
                g_cv.notify_all();
            } break;

            case SOURCE_PORT:{
                uint8_t buff[256];
                const int len = COMReceive(husb, 2, buff);
                if (len <= 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }
                port_buffer.append(reinterpret_cast<const char*>(buff), static_cast<size_t>(len));
                if (port_buffer.size() > 4096) {
                    port_buffer.erase(0, port_buffer.size() - 1024);
                }

                size_t pos = 0;
                while ((pos = port_buffer.find('\n')) != std::string::npos) {
                    std::string line = port_buffer.substr(0, pos);
                    port_buffer.erase(0, pos + 1);

                    if (!line.empty() && line.back() == '\r') {
                        line.pop_back();
                    }
                    if (line.empty()) {
                        continue;
                    }

                    double recv[7] = {0.0};
                    const int parsed = std::sscanf(
                        line.c_str(),
                        "%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                        &recv[0], &recv[1], &recv[2], &recv[3],
                        &recv[4], &recv[5], &recv[6]
                    );

                    if(parsed != 7){
                        Logger(WARN, "Parse IMU via port failed: %s", line.c_str());
                        continue;
                    }

                    std::lock_guard<std::mutex> lock(mut);
                    pushImuSampleLocked(recv);

                    gout.depth_tsms = 0.0;
                    gout.depth.release();
                    gout.frame_tsms = 0.0;
                    gout.frame_dtms = 0.0;
                    gout.frame.release();
                    source_state.store(SOURCEMAN_OK, std::memory_order_relaxed);
                    g_cv.notify_all();
                }
            }break;

            case SOURCE_RTSP:
            case SOURCE_MP4:{
                if(config->gen.type == SOURCE_MP4){
                    const double fps_out = (config->cam.fps > 0.0) ? config->cam.fps : 30.0;
                    const auto frame_period = std::chrono::duration<double>(1.0 / fps_out);

                    const auto now = std::chrono::steady_clock::now();
                    const auto elapsed = now - mp4_last_tick;
                    if (elapsed < frame_period) {
                        std::this_thread::sleep_for(frame_period - elapsed);
                    }
                    mp4_last_tick = std::chrono::steady_clock::now();
                }

                cv::Mat frame;
                if(!cap.read(frame) || frame.empty()){
                    if (config->gen.type == SOURCE_MP4) {
                        Logger(INFO, "MP4 reached EOF or frame decode failed");
                        source_state.store(SOURCEMAN_EOF, std::memory_order_relaxed);
                        g_cv.notify_all();
                    } else {
                        Logger(WARN, "RTSP frame read failed");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                cv::Mat out_frame;
                if(config->cam.w > 0 && config->cam.h > 0 &&
                (frame.cols != config->cam.w || frame.rows != config->cam.h)) {
                    cv::resize(frame, out_frame, cv::Size(config->cam.w, config->cam.h));
                } else {
                    out_frame = frame;
                }

                const double frame_ts_ms = cap.get(cv::CAP_PROP_POS_MSEC);
                SourceIn frame_pkt;
                frame_pkt.depth_tsms = 0.0;
                frame_pkt.depth.release();
                frame_pkt.frame_dtms = (old_frame_ts < 0.0) ? 0.0 : (frame_ts_ms - old_frame_ts);
                frame_pkt.frame_tsms = frame_ts_ms;
                frame_pkt.frame = out_frame.clone();
                old_frame_ts = frame_ts_ms;

                std::unique_lock<std::mutex> lock(mut);
                if(config->gen.type != SOURCE_MP4 && config->gen.imu_on == true){
                    receiveUdpImuLocked();
                }else{
                    gout.acc.clear();
                    gout.gyr.clear();
                    gout.acc_tsms.clear();
                    gout.gyr_tsms.clear();
                    gout.acc_dtms.clear();
                    gout.gyr_dtms.clear();
                }

                moveAccumulatorToPacketLocked(&frame_pkt);
                pushFramePacketLocked(lock, std::move(frame_pkt));
                source_state.store(SOURCEMAN_OK, std::memory_order_relaxed);
            }break;
        }
    }
}
bool initSourceMan(Config * config){
    run_polling.store(false, std::memory_order_relaxed);
    source_state.store(SOURCEMAN_OK, std::memory_order_relaxed);
    old_frame_ts = -1.0;
    resetRsTimestamps();
    {
        std::lock_guard<std::mutex> lock(mut);
        clearSourceOutLocked(&gout);
        g_frame_queue.clear();
    }
    g_cv.notify_all();
    g_bundle_imu_with_frame = (config != nullptr && config->gen.color_on && config->gen.imu_on);
    g_imu_only_mode = (config != nullptr && config->gen.imu_on && !config->gen.color_on);
    g_imu_batch_ms = (config != nullptr) ? std::max(0.0, config->imu.batch_ms) : 0.0;
    g_frame_packets_dropped = 0;
    g_rotate_imu_to_color = false;
    g_R_ci = mat3::Identity();
    if (config != nullptr &&
        !config->imu.T.empty() &&
        config->imu.T.rows >= 3 && config->imu.T.cols >= 3 &&
        (config->gen.type == SOURCE_BAG || config->gen.type == SOURCE_RSCAM)) {
        cv::Mat R_ci_cv = config->imu.T(cv::Range(0, 3), cv::Range(0, 3)).clone();
        R_ci_cv.convertTo(R_ci_cv, CV_64F);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                g_R_ci(r, c) = R_ci_cv.at<double>(r, c);
            }
        }
        g_rotate_imu_to_color = true;
    }

    switch(config->gen.type){
    case SOURCE_RSCAM: {
        rs2::config live_config;

        if (config->gen.groundt && config->gen.color_on && config->gen.imu_on) {
            live_config.enable_stream(RS2_STREAM_DEPTH);
            live_config.enable_stream(RS2_STREAM_COLOR);
            live_config.enable_stream(RS2_STREAM_ACCEL);
            live_config.enable_stream(RS2_STREAM_GYRO);

        } else if (!config->gen.groundt && config->gen.color_on && config->gen.imu_on) {
            live_config.enable_stream(RS2_STREAM_COLOR);
            live_config.enable_stream(RS2_STREAM_ACCEL);
            live_config.enable_stream(RS2_STREAM_GYRO);

        } else if (!config->gen.groundt && !config->gen.color_on && config->gen.imu_on) {
            live_config.enable_stream(RS2_STREAM_ACCEL);
            live_config.enable_stream(RS2_STREAM_GYRO);

        } else if (!config->gen.groundt && config->gen.color_on && !config->gen.imu_on) {
            live_config.enable_stream(RS2_STREAM_COLOR);

        } else if (config->gen.groundt && config->gen.color_on && !config->gen.imu_on) {
            live_config.enable_stream(RS2_STREAM_DEPTH);
            live_config.enable_stream(RS2_STREAM_COLOR);

        } else if (config->gen.groundt && !config->gen.color_on && config->gen.imu_on) {
            live_config.enable_stream(RS2_STREAM_DEPTH);
            live_config.enable_stream(RS2_STREAM_ACCEL);
            live_config.enable_stream(RS2_STREAM_GYRO);

        } else if (config->gen.groundt && !config->gen.color_on && !config->gen.imu_on) {
            live_config.enable_stream(RS2_STREAM_DEPTH);

        } else {
            Logger(ERROR, "At least one stream must be selected for RS camera");
            return false;
        }

        rs2::pipeline_profile profile = pipe.start(
            live_config,
            [&](const rs2::frame& f) { queue.enqueue(f); }
        );

        if (config->gen.groundt && config->gen.color_on) {
            delete g_align_to_color;
            g_align_to_color = new rs2::align(RS2_STREAM_COLOR);
        }

        if (profile.get_device().is<rs2::playback>()) {
            rs2::playback playback = profile.get_device().as<rs2::playback>();
            playback.set_real_time(false);
            Logger(INFO, "Bag playback set to non-real-time");
        }

        fillConfigFromProfile(profile, config);

        g_source_man_init = true;
        run_polling.store(true, std::memory_order_relaxed);
        pollThread = std::thread(pollingThread, config);
        return true;
    } break;

    case SOURCE_BAG: {
        rs2::config bag_config;
        bag_config.enable_device_from_file(config->gen.input, false);

        if (config->gen.groundt && config->gen.color_on && config->gen.imu_on) {
            bag_config.enable_stream(RS2_STREAM_DEPTH);
            bag_config.enable_stream(RS2_STREAM_COLOR);
            bag_config.enable_stream(RS2_STREAM_ACCEL);
            bag_config.enable_stream(RS2_STREAM_GYRO);
            Logger(INFO, "Init with imu, depth and color");
        } else if (!config->gen.groundt && config->gen.color_on && config->gen.imu_on) {
            bag_config.enable_stream(RS2_STREAM_COLOR);
            bag_config.enable_stream(RS2_STREAM_ACCEL);
            bag_config.enable_stream(RS2_STREAM_GYRO);
            Logger(INFO, "Init with imu and color");
        } else if (!config->gen.groundt && !config->gen.color_on && config->gen.imu_on) {
            bag_config.enable_stream(RS2_STREAM_ACCEL);
            bag_config.enable_stream(RS2_STREAM_GYRO);
            Logger(INFO, "Init with imu");
        } else if (!config->gen.groundt && config->gen.color_on && !config->gen.imu_on) {
            bag_config.enable_stream(RS2_STREAM_COLOR);
            Logger(INFO, "Init with color");
        } else if (config->gen.groundt && config->gen.color_on && !config->gen.imu_on) {
            bag_config.enable_stream(RS2_STREAM_DEPTH);
            bag_config.enable_stream(RS2_STREAM_COLOR);
            Logger(INFO, "Init with depth and color");
        } else if (config->gen.groundt && !config->gen.color_on && config->gen.imu_on) {
            bag_config.enable_stream(RS2_STREAM_DEPTH);
            bag_config.enable_stream(RS2_STREAM_ACCEL);
            bag_config.enable_stream(RS2_STREAM_GYRO);
            Logger(INFO, "Init with depth and IMU");
        } else if (config->gen.groundt && !config->gen.color_on && !config->gen.imu_on) {
            bag_config.enable_stream(RS2_STREAM_DEPTH);
            Logger(INFO, "Init with depth");
        } else {
            Logger(ERROR, "A stream must be selected for bag file");
            return false;
        }

        rs2::pipeline_profile profile = pipe.start(
            bag_config,
            [&](const rs2::frame& f) { queue.enqueue(f); }
        );

        if (config->gen.groundt && config->gen.color_on) {
            delete g_align_to_color;
            g_align_to_color = new rs2::align(RS2_STREAM_COLOR);
        }

        fillConfigFromProfile(profile, config);

        g_source_man_init = true;
        run_polling.store(true, std::memory_order_relaxed);
        pollThread = std::thread(pollingThread, config);
        return true;
    } break;

    case SOURCE_MP4:
    case SOURCE_RTSP:{
        if(!cap.open(config->gen.input)){
            Logger(ERROR, "initSourceMan => Source %s could not be opened", config->gen.input.c_str());
            return false;
        }

        const double src_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const double src_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double real_fps = cap.get(cv::CAP_PROP_FPS);

        if(real_fps <= 0.0 || std::isnan(real_fps)){
            real_fps = 30.0;
            Logger(WARN, "Could not read source FPS, using fallback => %f", real_fps);
        }


        if(config->cam.fps <= 0.0){
            config->cam.fps = static_cast<float>(real_fps);
        }else if(config->cam.fps >= real_fps){
            config->cam.fps = static_cast<float>(real_fps);
        }

        if(config->cam.w <= 0){
            config->cam.w = static_cast<int>(src_w);
        }
        if(config->cam.h <= 0){
            config->cam.h = static_cast<int>(src_h);
        }

        if(config->gen.type == SOURCE_RTSP && config->gen.imu_on){
            if(!UDPInit(&hudp)){
                Logger(ERROR, "initSourceMan => UDP IMU source could not be opened");
                cap.release();
                return false;
            }

            u_long non_blocking = 1;
            ioctlsocket(hudp, FIONBIO, &non_blocking);

            const int recv_buff_size = 1 << 20;
            setsockopt(hudp, SOL_SOCKET, SO_RCVBUF,
                reinterpret_cast<const char*>(&recv_buff_size), sizeof(recv_buff_size));
        }

        config->cam.is_rgb = true;

        Logger(INFO, "Video source opened. Source size=%dx%d source fps=%.3f output size=%dx%d output fps=%.3f",
            static_cast<int>(src_w), static_cast<int>(src_h), real_fps,
            config->cam.w, config->cam.h, config->cam.fps);

        g_source_man_init = true;
        run_polling.store(true, std::memory_order_relaxed);
        pollThread = std::thread(pollingThread, config);
        return true;
    }break;

    case SOURCE_CSV:{
        rheader.clear();
        bool ok = reader.init(config->gen.input.c_str(), &rheader);
        if(!ok){
            Logger(ERROR, "initSourceMan => Source %s could not be opened", config->gen.input.c_str());
            return false;
        }

        g_source_man_init = true;
        run_polling.store(true, std::memory_order_relaxed);
        pollThread = std::thread(pollingThread, config);
        return true;
    }break;

    case SOURCE_PORT:{
        std::wstring widestr(config->gen.input.begin(), config->gen.input.end());
        if (!COMInit(husb, widestr.c_str())) {
            Logger(ERROR, "initSourceMan => Source %s could not be opened", config->gen.input.c_str());
            return false;
        }

        g_source_man_init = true;
        run_polling.store(true, std::memory_order_relaxed);
        pollThread = std::thread(pollingThread, config);
        return true;

    }break;

    default:{
        Logger(ERROR, "Source is not in the lsit of available");
        return false;
    }break;
    }
}
void closeSourceMan(){
    run_polling.store(false, std::memory_order_relaxed);

    if(pollThread.joinable()){
        pollThread.join();
    }

    if (husb != INVALID_HANDLE_VALUE) {
        COMDeInit(husb);
        husb = INVALID_HANDLE_VALUE;
    }
    if (hudp != INVALID_SOCKET) {
        UDPDeInit(&hudp);
        hudp = INVALID_SOCKET;
    }

    try{
        pipe.stop();
    }catch (...) {}

    if (g_align_to_color != nullptr) {
        delete g_align_to_color;
        g_align_to_color = nullptr;
    }

    if(cap.isOpened()){
        cap.release();
    }

    source_state.store(SOURCEMAN_OK, std::memory_order_relaxed);
    g_source_man_init = false;
    g_bundle_imu_with_frame = false;
    g_imu_only_mode = false;
    g_imu_batch_ms = 0.0;
    {
        std::lock_guard<std::mutex> lock(mut);
        clearSourceOutLocked(&gout);
        g_frame_queue.clear();
    }
    g_cv.notify_all();
}
SourceManType getSourceMan(SourceIn * out){
    if(!g_source_man_init || out == nullptr)return SOURCEMAN_ERR;

    std::unique_lock<std::mutex> lock(mut);
    clearSourceOutLocked(out);

    if (g_imu_only_mode) {
        g_cv.wait(lock, [] {
            const SourceManType st = source_state.load(std::memory_order_relaxed);
            return imuBatchReadyLocked() ||
                   st == SOURCEMAN_EOF ||
                   st == SOURCEMAN_ERR ||
                   !run_polling.load(std::memory_order_relaxed);
        });

        if (imuBatchReadyLocked()) {
            out->acc = gout.acc;
            out->acc_tsms = gout.acc_tsms;
            out->acc_dtms = gout.acc_dtms;
            out->gyr = gout.gyr;
            out->gyr_tsms = gout.gyr_tsms;
            out->gyr_dtms = gout.gyr_dtms;

            std::deque<vec3> next_acc;
            std::deque<double> next_acc_tsms;
            std::deque<vec3> next_gyr;
            std::deque<double> next_gyr_tsms;

            next_acc.emplace_back(gout.acc.back());
            next_acc_tsms.emplace_back(gout.acc_tsms.back());
            next_gyr.emplace_back(gout.gyr.back());
            next_gyr_tsms.emplace_back(gout.gyr_tsms.back());

            gout.acc.swap(next_acc);
            gout.acc_tsms.swap(next_acc_tsms);
            gout.acc_dtms.clear();
            gout.gyr.swap(next_gyr);
            gout.gyr_tsms.swap(next_gyr_tsms);
            gout.gyr_dtms.clear();

            // Logger(DEBUG,
            //     "SourceMan IMU_ONLY pop acc=%zu gyr=%zu t0=%.3f tf=%.3f span=%.3fms",
            //     out->acc.size(), out->gyr.size(),
            //     out->gyr_tsms.empty() ? 0.0 : out->gyr_tsms.front(),
            //     out->gyr_tsms.empty() ? 0.0 : out->gyr_tsms.back(),
            //     out->gyr_tsms.size() >= 2 ? (out->gyr_tsms.back() - out->gyr_tsms.front()) : 0.0);
            return SOURCEMAN_OK;
        }

        return source_state.load(std::memory_order_relaxed);
    }

    if (g_bundle_imu_with_frame) {
        g_cv.wait(lock, [] {
            const SourceManType st = source_state.load(std::memory_order_relaxed);
            return !g_frame_queue.empty() || st == SOURCEMAN_EOF || st == SOURCEMAN_ERR || !run_polling.load(std::memory_order_relaxed);
        });

        if (!g_frame_queue.empty()) {
            *out = std::move(g_frame_queue.front());
            g_frame_queue.pop_front();
            g_cv.notify_all();
            Logger(DEBUG,
                "SourceMan FRAME pop ts=%.3f acc=%zu gyr=%zu q=%zu",
                out->frame_tsms, out->acc.size(), out->gyr.size(), g_frame_queue.size());
            return SOURCEMAN_OK;
        }

        
        return source_state.load(std::memory_order_relaxed);
    }

    if (!g_frame_queue.empty()) {
        *out = std::move(g_frame_queue.front());
        g_frame_queue.pop_front();
        g_cv.notify_all();
        return SOURCEMAN_OK;
    }

    if (gout.acc.size() >= 2 && gout.gyr.size() >= 2) {
        out->acc = gout.acc;
        out->acc_tsms = gout.acc_tsms;
        out->acc_dtms = gout.acc_dtms;
        out->gyr = gout.gyr;
        out->gyr_tsms = gout.gyr_tsms;
        out->gyr_dtms = gout.gyr_dtms;

        std::deque<vec3> next_acc;
        std::deque<double> next_acc_tsms;
        std::deque<vec3> next_gyr;
        std::deque<double> next_gyr_tsms;

        next_acc.emplace_back(gout.acc.back());
        next_acc_tsms.emplace_back(gout.acc_tsms.back());
        next_gyr.emplace_back(gout.gyr.back());
        next_gyr_tsms.emplace_back(gout.gyr_tsms.back());

        gout.acc.swap(next_acc);
        gout.acc_tsms.swap(next_acc_tsms);
        gout.acc_dtms.clear();
        gout.gyr.swap(next_gyr);
        gout.gyr_tsms.swap(next_gyr_tsms);
        gout.gyr_dtms.clear();

        return SOURCEMAN_OK;
    }

    return source_state.load(std::memory_order_relaxed);
}



static void fillConfigFromProfile(const rs2::pipeline_profile& profile, Config* config)
{
    if (config == nullptr) {
        return;
    }

    for (const rs2::stream_profile& sp : profile.get_streams()) {
        const rs2_stream st = sp.stream_type();

        if (st == RS2_STREAM_COLOR) {
            rs2::video_stream_profile vsp = sp.as<rs2::video_stream_profile>();
            if (vsp) {
                config->cam.w   = static_cast<uint32_t>(vsp.width());
                config->cam.h   = static_cast<uint32_t>(vsp.height());
                if(config->cam.fps > static_cast<double>(vsp.fps())){
                    config->cam.fps = static_cast<double>(vsp.fps());
                }

                const rs2_format fmt = vsp.format();
                if (fmt == RS2_FORMAT_RGB8 || fmt == RS2_FORMAT_RGBA8) {
                    config->cam.is_rgb = true;
                }
                else if (fmt == RS2_FORMAT_BGR8 || fmt == RS2_FORMAT_BGRA8) {
                    config->cam.is_rgb = false;
                }
            }
        }
        else if (st == RS2_STREAM_DEPTH) {
            continue;
        }
        else if (st == RS2_STREAM_ACCEL) {
            config->imu.acc_fps = static_cast<double>(sp.fps());
        }
        else if (st == RS2_STREAM_GYRO) {
            config->imu.gyr_fps = static_cast<double>(sp.fps());
        }
    }
}
