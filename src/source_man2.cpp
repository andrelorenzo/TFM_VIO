// #include "source_man.hpp"
#include "config.hpp"
#include "csv_logger.hpp"
#include "runtime_control.hpp"

// seconds
#include "seconds/comms_common.h"


// thirds
#include "librealsense2/rs.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <thread>
#include <climits>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <cmath>
#include <utility>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <sstream>

#define MAX_HIST 5000
#define MAX_RTSP_IMU_BUF 5000
#define MAX_RTSP_STAMP_MAP 512

#define MSGID_ESTABLISH_CONNECTION 0x0001
#define MSGID_UNSESTABLISH_CONNECTION 0x0002
#define MSGID_ACCEL_DATA 0x0011
#define MSGID_GYRO_DATA 0x0012

static rs2::pipeline pipe;
static rs2::frame_queue queue(MAX_HIST * 2);
static rs2::align* g_align_to_color = nullptr;
static std::unique_ptr<rs2::playback> g_bag_playback;
static std::thread poll_thread;
static HANDLE husb = INVALID_HANDLE_VALUE;

static std::mutex dout_mutex;
static std::deque<SourceIn> vout;

static Config source_config;
static bool source_open = false;
static std::atomic<bool> source_finished(false);
static std::atomic<bool> source_stop_requested(false);
static std::vector<ImuSample> csv_imu_samples;
static size_t csv_next_sample = 0;

static void updateSource2(Config * config);
static void recvRtspImu(uint8_t *msg, size_t len, const char *ip, uint16_t port, uint16_t cid);
static bool loadCsvImuSamples(const std::string& path, Config * config);
static bool pollCsv();
void closeSource2();

#pragma pack(push, 1)
struct RtspAccelMsg {
    uint32_t seq;
    uint64_t ts_us;
    uint32_t ts_domain;
    int32_t acc[3];
};

struct RtspGyroMsg {
    uint32_t seq;
    uint64_t ts_us;
    uint32_t ts_domain;
    int32_t gyro[3];
};
#pragma pack(pop)

struct RtspVideoStamp {
    uint64_t camera_ts_us = 0;
    uint32_t seq = 0;
    uint32_t ts_domain = 0;
};

class TimestampNormalizer {
public:
    void reset() {
        init_ = false;
        first_raw_ms_ = 0.0;
        last_raw_ms_ = 0.0;
        wrap_offset_ms_ = 0.0;
    }

    double fromMs(double raw_ts_ms) {
        const double WRAP_THRESHOLD_MS = 1000.0;

        if (!init_) {
            init_ = true;
            first_raw_ms_ = raw_ts_ms;
            last_raw_ms_ = raw_ts_ms;
            wrap_offset_ms_ = 0.0;
            return 0.0;
        }

        if (raw_ts_ms + WRAP_THRESHOLD_MS < last_raw_ms_) wrap_offset_ms_ += last_raw_ms_;
        last_raw_ms_ = raw_ts_ms;

        return wrap_offset_ms_ + raw_ts_ms - first_raw_ms_;
    }

    double fromUs(uint64_t raw_ts_us) {
        return fromMs(static_cast<double>(raw_ts_us) / 1000.0);
    }

private:
    bool init_ = false;
    double first_raw_ms_ = 0.0;
    double last_raw_ms_ = 0.0;
    double wrap_offset_ms_ = 0.0;
};

static std::mutex rtsp_ts_mutex;
static TimestampNormalizer rtsp_ts_norm;

static std::mutex rtsp_imu_mutex;
static std::deque<std::pair<double, vec3>> rtsp_acc_buf;
static std::deque<std::pair<double, vec3>> rtsp_gyr_buf;

static std::mutex rtsp_stamp_mutex;
static std::map<guint64, RtspVideoStamp> rtsp_video_stamps;

static std::mutex rtsp_udp_send_mutex;
static std::atomic<bool> rtsp_udp_run(false);
static std::thread rtsp_udp_thread;
static std::thread rtsp_udp_recv_thread;
static std::string rtsp_server_host;
static std::wstring rtsp_server_host_w;

static std::mutex rtsp_gst_mutex;
static GstElement *rtsp_pipeline = nullptr;
static GstElement *rtsp_appsink = nullptr;

static SOCKET hsocket = INVALID_SOCKET;


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

static double normRtspTsUs(uint64_t raw_ts_us) {
    std::lock_guard<std::mutex> lock(rtsp_ts_mutex);
    return rtsp_ts_norm.fromUs(raw_ts_us);
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

static void insertSortedImu(std::deque<std::pair<double, vec3>>& buf, double ts_ms, const vec3& value) {
    if (buf.empty() || ts_ms >= buf.back().first) {
        buf.emplace_back(ts_ms, value);
    }
    else {
        auto it = std::lower_bound(buf.begin(), buf.end(), ts_ms, [](const std::pair<double, vec3>& sample, double ts) { return sample.first < ts; });
        if (it == buf.end() || std::abs(it->first - ts_ms) > 1e-9) buf.insert(it, std::make_pair(ts_ms, value));
    }

    while (buf.size() > MAX_RTSP_IMU_BUF) buf.pop_front();
}

static std::string hostFromRtspUrl(const std::string& url) {
    const std::string prefix = "rtsp://";
    size_t begin = 0;

    if (url.compare(0, prefix.size(), prefix) == 0) begin = prefix.size();

    const size_t slash = url.find('/', begin);
    std::string authority = url.substr(begin, slash == std::string::npos ? std::string::npos : slash - begin);

    const size_t at = authority.find('@');
    if (at != std::string::npos) authority = authority.substr(at + 1);

    if (!authority.empty() && authority[0] == '[') {
        const size_t end_bracket = authority.find(']');
        if (end_bracket != std::string::npos) return authority.substr(1, end_bracket - 1);
    }

    const size_t colon = authority.find(':');
    if (colon != std::string::npos) authority = authority.substr(0, colon);

    if (authority.empty()) authority = RTSP_IP;
    return authority;
}

static std::wstring widenAscii(const std::string& value) {
    return std::wstring(value.begin(), value.end());
}

static bool isIpv4Literal(const std::wstring& value) {
    sockaddr_in addr = {};
    return InetPtonW(AF_INET, value.c_str(), &addr.sin_addr) == 1;
}

static std::wstring getExecutableDirW() {
    wchar_t path[MAX_PATH] = {};
    const DWORD len = GetModuleFileNameW(nullptr, path, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) return L"";

    std::wstring dir(path, len);
    const size_t slash = dir.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return L"";
    dir.resize(slash);
    return dir;
}

static void setupBundledGStreamerRuntime() {
    const std::wstring exe_dir = getExecutableDirW();
    if (exe_dir.empty()) return;

    const std::wstring plugin_dir = exe_dir + L"\\gstreamer-1.0";
    const std::wstring scanner_path = exe_dir + L"\\libexec\\gstreamer-1.0\\gst-plugin-scanner.exe";

    SetEnvironmentVariableW(L"GST_PLUGIN_PATH", plugin_dir.c_str());
    SetEnvironmentVariableW(L"GST_PLUGIN_PATH_1_0", plugin_dir.c_str());
    SetEnvironmentVariableW(L"GST_PLUGIN_SYSTEM_PATH_1_0", plugin_dir.c_str());
    SetEnvironmentVariableW(L"GST_PLUGIN_SCANNER", scanner_path.c_str());

    DWORD current_len = GetEnvironmentVariableW(L"PATH", nullptr, 0);
    std::wstring current_path;
    if (current_len > 0) {
        current_path.resize(current_len);
        const DWORD copied = GetEnvironmentVariableW(L"PATH", &current_path[0], current_len);
        if (copied > 0 && copied < current_len) current_path.resize(copied);
        else current_path.clear();
    }

    if (current_path.find(exe_dir) == std::wstring::npos) {
        SetEnvironmentVariableW(L"PATH", (exe_dir + L";" + current_path).c_str());
    }
}

static int sendRtspUdpRaw(const std::wstring& ip, uint16_t port, const uint8_t* msg, size_t msg_len) {
    if (hsocket == INVALID_SOCKET || msg == nullptr || msg_len == 0 || ip.empty() || port == 0) return -1;

    sockaddr_in dest = {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(port);

    if (InetPtonW(AF_INET, ip.c_str(), &dest.sin_addr) != 1) return -2;

    const int to_send = msg_len > static_cast<size_t>(INT_MAX) ? INT_MAX : static_cast<int>(msg_len);
    return sendto(hsocket, reinterpret_cast<const char*>(msg), to_send, 0, reinterpret_cast<const sockaddr*>(&dest), sizeof(dest));
}

static bool sendRtspUdpPayload(uint16_t msg_id, const void *data, size_t data_size) {
    if (hsocket == INVALID_SOCKET || rtsp_server_host_w.empty()) return false;

    comms_payload_t pay;
    uint8_t frame[4096 * 2];
    Crc32 crc;
    uint32_t keepalive_dummy = 0;

    memset(&pay, 0, sizeof(pay));

    if (data_size > sizeof(pay.data)) {
        Logger(ERROR, "RTSP UDP payload too large: %zu bytes", data_size);
        return false;
    }

    // El protocolo practico que usa VISLAM con Project_commons.lib trabaja
    // siempre con payloads alineados a 4 bytes. El start/stop no lleva datos,
    // asi que enviamos 4 bytes dummy que el receptor ignora.
    const void *payload_ptr = data;
    size_t payload_size = data_size;
    if (payload_size == 0) {
        payload_ptr = &keepalive_dummy;
        payload_size = sizeof(keepalive_dummy);
    }

    pay.msg_id = msg_id;
    pay.data_size = static_cast<uint16_t>(payload_size);
    memcpy(pay.data, payload_ptr, payload_size);

    pay.crc_32 = crc.Calculate(&pay);

    const int frame_len = EncodeFrame(&pay, frame);
    if (frame_len <= 0) {
        Logger(WARN, "RTSP UDP frame no pudo codificarse msg_id=0x%04X payload_size=%zu", msg_id, payload_size);
        return false;
    }

    std::lock_guard<std::mutex> lock(rtsp_udp_send_mutex);
    const int sent = sendRtspUdpRaw(rtsp_server_host_w, UDP_SEND_PORT, frame, static_cast<size_t>(frame_len));
    if (sent == SOCKET_ERROR) {
        Logger(WARN, "RTSP UDP sendto(%S:%u) fallo con WSA error=%d", rtsp_server_host_w.c_str(), UDP_SEND_PORT, WSAGetLastError());
        return false;
    }
    if (msg_id == MSGID_ESTABLISH_CONNECTION || msg_id == MSGID_UNSESTABLISH_CONNECTION) {
        Logger(INFO, "RTSP UDP msg_id=0x%04X enviado a %S:%u frame_len=%d", msg_id, rtsp_server_host_w.c_str(), UDP_SEND_PORT, frame_len);
    }
    return sent > 0;
}

static void rtspUdpKeepaliveLoop() {
    while (rtsp_udp_run.load()) {
        sendRtspUdpPayload(MSGID_ESTABLISH_CONNECTION, nullptr, 0);

        for (int i = 0; i < 100 && rtsp_udp_run.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

static void rtspUdpRecvLoop() {
    uint8_t frame[BUFFER_DEFAULT_SIZE * 2] = {};

    while (rtsp_udp_run.load()) {
        const int recv_len = UDPReceive(hsocket, frame, sizeof(frame));

        if (recv_len > 0) {
            recvRtspImu(frame, static_cast<size_t>(recv_len), nullptr, 0, 0);
        }
        else if (recv_len < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

static void recvRtspImu(uint8_t *msg, size_t len, const char *ip, uint16_t port, uint16_t cid) {
    (void)ip;
    (void)port;

    if (len <= 0 || cid != 0) return;

    Crc32 crc;
    comms_payload_t pay;
    memset(&pay, 0, sizeof(pay));

    if (DecodeFrame(msg, static_cast<int>(len), &pay) < 0) return;

    if (!crc.Check(&pay)) return;

    if (pay.msg_id == MSGID_ACCEL_DATA) {
        if (pay.data_size < sizeof(RtspAccelMsg)) return;

        RtspAccelMsg m;
        memcpy(&m, pay.data, sizeof(m));

        const double ts_ms = normRtspTsUs(m.ts_us);
        const vec3 acc(static_cast<double>(m.acc[0]) / 1000000.0, static_cast<double>(m.acc[1]) / 1000000.0, static_cast<double>(m.acc[2]) / 1000000.0);

        std::lock_guard<std::mutex> lock(rtsp_imu_mutex);
        insertSortedImu(rtsp_acc_buf, ts_ms, acc);
    }
    else if (pay.msg_id == MSGID_GYRO_DATA) {
        if (pay.data_size < sizeof(RtspGyroMsg)) return;

        RtspGyroMsg m;
        memcpy(&m, pay.data, sizeof(m));

        const double ts_ms = normRtspTsUs(m.ts_us);
        const vec3 gyr(static_cast<double>(m.gyro[0]) / 1000000.0, static_cast<double>(m.gyro[1]) / 1000000.0, static_cast<double>(m.gyro[2]) / 1000000.0);

        std::lock_guard<std::mutex> lock(rtsp_imu_mutex);
        insertSortedImu(rtsp_gyr_buf, ts_ms, gyr);
    }
}

static void closeRtspUdp() {
    if (hsocket != INVALID_SOCKET && rtsp_server_host_w.size() > 0) {
        sendRtspUdpPayload(MSGID_UNSESTABLISH_CONNECTION, nullptr, 0);
    }

    rtsp_udp_run.store(false);

    if (rtsp_udp_thread.joinable()) rtsp_udp_thread.join();
    if (rtsp_udp_recv_thread.joinable()) rtsp_udp_recv_thread.join();

    if (hsocket != INVALID_SOCKET) {
        UDPDeInit(&hsocket);
        hsocket = INVALID_SOCKET;
    }
}

static bool initRtspUdp(const std::string& host) {
    closeRtspUdp();

    rtsp_server_host = host;
    rtsp_server_host_w = widenAscii(host);
    if (!isIpv4Literal(rtsp_server_host_w)) {
        Logger(WARN, "RTSP host '%s' no es una IPv4 literal para el canal UDP; usando %S", host.c_str(), UDP_IP);
        rtsp_server_host = RTSP_IP;
        rtsp_server_host_w = UDP_IP;
    }

    if(!UDPInit(&hsocket)) return false;

    {
        std::lock_guard<std::mutex> lock(rtsp_imu_mutex);
        rtsp_acc_buf.clear();
        rtsp_gyr_buf.clear();
    }

    {
        std::lock_guard<std::mutex> lock(rtsp_ts_mutex);
        rtsp_ts_norm.reset();
    }

    rtsp_udp_run.store(true);

    if (!sendRtspUdpPayload(MSGID_ESTABLISH_CONNECTION, nullptr, 0)) {
        Logger(WARN, "RTSP UDP start message could not be sent to %s:%u", rtsp_server_host.c_str(), UDP_SEND_PORT);
    }

    rtsp_udp_thread = std::thread(rtspUdpKeepaliveLoop);
    rtsp_udp_recv_thread = std::thread(rtspUdpRecvLoop);

    Logger(INFO, "RTSP IMU UDP client: local_port=%u remote=%s:%u", UDP_RECV_PORT, rtsp_server_host.c_str(), UDP_SEND_PORT);
    return true;
}

static uint32_t readBe32(const uint8_t *p) {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) |
           static_cast<uint32_t>(p[3]);
}

static uint64_t readBe64(const uint8_t *p) {
    return (static_cast<uint64_t>(p[0]) << 56) |
           (static_cast<uint64_t>(p[1]) << 48) |
           (static_cast<uint64_t>(p[2]) << 40) |
           (static_cast<uint64_t>(p[3]) << 32) |
           (static_cast<uint64_t>(p[4]) << 24) |
           (static_cast<uint64_t>(p[5]) << 16) |
           (static_cast<uint64_t>(p[6]) << 8) |
           static_cast<uint64_t>(p[7]);
}

static size_t startCodeLenAt(const uint8_t *data, size_t size, size_t pos) {
    if (pos + 3 <= size && data[pos] == 0x00 && data[pos + 1] == 0x00 && data[pos + 2] == 0x01) return 3;
    if (pos + 4 <= size && data[pos] == 0x00 && data[pos + 1] == 0x00 && data[pos + 2] == 0x00 && data[pos + 3] == 0x01) return 4;
    return 0;
}

static size_t findNextStartCode(const uint8_t *data, size_t size, size_t pos) {
    for (size_t i = pos; i + 3 < size; ++i) {
        if (data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x01) return i;
        if (i + 4 < size && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x00 && data[i + 3] == 0x01) return i;
    }

    return size;
}

static std::vector<uint8_t> nalToRbsp(const uint8_t *data, size_t size) {
    std::vector<uint8_t> rbsp;
    rbsp.reserve(size);

    int zero_count = 0;

    for (size_t i = 0; i < size; ++i) {
        const uint8_t b = data[i];

        if (zero_count >= 2 && b == 0x03) {
            zero_count = 0;
            continue;
        }

        rbsp.push_back(b);

        if (b == 0x00) zero_count++;
        else zero_count = 0;
    }

    return rbsp;
}

static bool parseRealSenseSeiRbsp(const std::vector<uint8_t>& rbsp, RtspVideoStamp& out) {
    static const uint8_t uuid[16] = {0x2D, 0x4F, 0x31, 0x8A, 0xB6, 0x34, 0x4F, 0x72, 0x9A, 0x42, 0x52, 0x53, 0x54, 0x53, 0x00, 0x01};

    size_t i = 0;

    while (i + 1 < rbsp.size()) {
        uint32_t payload_type = 0;

        while (i < rbsp.size() && rbsp[i] == 0xFF) {
            payload_type += 255;
            ++i;
        }

        if (i >= rbsp.size()) break;
        payload_type += rbsp[i++];

        uint32_t payload_size = 0;

        while (i < rbsp.size() && rbsp[i] == 0xFF) {
            payload_size += 255;
            ++i;
        }

        if (i >= rbsp.size()) break;
        payload_size += rbsp[i++];

        if (i + payload_size > rbsp.size()) break;

        const uint8_t *payload = rbsp.data() + i;

        if (payload_type == 5 && payload_size >= 16 + 8 + 4 + 4 && memcmp(payload, uuid, 16) == 0) {
            out.camera_ts_us = readBe64(payload + 16);
            out.seq = readBe32(payload + 24);
            out.ts_domain = readBe32(payload + 28);
            return out.camera_ts_us > 0;
        }

        i += payload_size;
    }

    return false;
}

static bool parseRealSenseSeiH264(const uint8_t *data, size_t size, RtspVideoStamp& out) {
    size_t pos = 0;

    while (pos < size) {
        const size_t sc_len = startCodeLenAt(data, size, pos);

        if (sc_len == 0) {
            ++pos;
            continue;
        }

        const size_t nal_start = pos + sc_len;
        if (nal_start >= size) break;

        const size_t next = findNextStartCode(data, size, nal_start + 1);
        const uint8_t nal_type = data[nal_start] & 0x1F;

        if (nal_type == 6 && next > nal_start + 1) {
            const std::vector<uint8_t> rbsp = nalToRbsp(data + nal_start + 1, next - nal_start - 1);
            if (parseRealSenseSeiRbsp(rbsp, out)) return true;
        }

        pos = next;
    }

    return false;
}

static void registerRtspVideoStamp(guint64 pts, const RtspVideoStamp& stamp) {
    if (!GST_CLOCK_TIME_IS_VALID(pts) || stamp.camera_ts_us == 0) return;

    std::lock_guard<std::mutex> lock(rtsp_stamp_mutex);
    rtsp_video_stamps[pts] = stamp;
    while (rtsp_video_stamps.size() > MAX_RTSP_STAMP_MAP) rtsp_video_stamps.erase(rtsp_video_stamps.begin());
}

static bool takeRtspVideoStamp(guint64 pts, RtspVideoStamp& stamp) {
    std::lock_guard<std::mutex> lock(rtsp_stamp_mutex);

    if (rtsp_video_stamps.empty()) return false;

    if (GST_CLOCK_TIME_IS_VALID(pts)) {
        auto exact = rtsp_video_stamps.find(pts);
        if (exact != rtsp_video_stamps.end()) {
            stamp = exact->second;
            rtsp_video_stamps.erase(exact);
            return true;
        }

        auto upper = rtsp_video_stamps.lower_bound(pts);
        auto best = rtsp_video_stamps.end();
        guint64 best_delta = static_cast<guint64>(-1);

        if (upper != rtsp_video_stamps.end()) {
            const guint64 d = upper->first > pts ? upper->first - pts : pts - upper->first;
            best = upper;
            best_delta = d;
        }

        if (upper != rtsp_video_stamps.begin()) {
            auto prev = upper;
            --prev;
            const guint64 d = prev->first > pts ? prev->first - pts : pts - prev->first;
            if (d < best_delta) {
                best = prev;
                best_delta = d;
            }
        }

        if (best != rtsp_video_stamps.end() && best_delta < 200000000ULL) {
            stamp = best->second;
            rtsp_video_stamps.erase(best);
            return true;
        }
    }

    auto it = rtsp_video_stamps.begin();
    stamp = it->second;
    rtsp_video_stamps.erase(it);
    return true;
}

static GstPadProbeReturn rtspH264Probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data) {
    (void)pad;
    (void)user_data;

    if (!(GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER)) return GST_PAD_PROBE_OK;

    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READ)) return GST_PAD_PROBE_OK;

    RtspVideoStamp stamp;
    const bool ok = parseRealSenseSeiH264(map.data, map.size, stamp);

    gst_buffer_unmap(buf, &map);

    if (ok) registerRtspVideoStamp(GST_BUFFER_PTS(buf), stamp);

    return GST_PAD_PROBE_OK;
}

static bool rtspPipelineAvailable() {
    std::lock_guard<std::mutex> lock(rtsp_gst_mutex);
    return rtsp_pipeline != nullptr;
}

static void closeRtspPipeline() {
    std::lock_guard<std::mutex> lock(rtsp_gst_mutex);

    if (rtsp_pipeline) gst_element_set_state(rtsp_pipeline, GST_STATE_NULL);

    if (rtsp_appsink) {
        gst_object_unref(rtsp_appsink);
        rtsp_appsink = nullptr;
    }

    if (rtsp_pipeline) {
        gst_object_unref(rtsp_pipeline);
        rtsp_pipeline = nullptr;
    }

    {
        std::lock_guard<std::mutex> stamp_lock(rtsp_stamp_mutex);
        rtsp_video_stamps.clear();
    }
}

static bool initRtspPipeline(const std::string& url) {
    closeRtspPipeline();

    if (gst_element_factory_find("rtspsrc") == nullptr) {
        Logger(ERROR, "GStreamer no encuentra 'rtspsrc'. Verifica que los plugins esten copiados y accesibles desde el ejecutable.");
        return false;
    }

    GError *err = nullptr;

    const std::string pipe_desc =
        "rtspsrc location=\"" + url + "\" latency=0 protocols=tcp drop-on-latency=true ! "
        "rtph264depay ! "
        "h264parse name=rs_h264parse ! "
        "video/x-h264,stream-format=byte-stream,alignment=au ! "
        "queue ! decodebin ! videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink name=rs_sink emit-signals=false sync=false max-buffers=2 drop=true";

    GstElement *pipeline = gst_parse_launch(pipe_desc.c_str(), &err);

    if (!pipeline) {
        Logger(ERROR, "RTSP pipeline creation failed: %s", err ? err->message : "unknown error");
        if (err) g_error_free(err);
        return false;
    }

    if (err) {
        Logger(WARN, "RTSP pipeline warning: %s", err->message);
        g_error_free(err);
    }

    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "rs_sink");
    GstElement *h264parse = gst_bin_get_by_name(GST_BIN(pipeline), "rs_h264parse");

    if (!appsink || !h264parse) {
        Logger(ERROR, "RTSP pipeline: appsink or h264parse not found");
        if (appsink) gst_object_unref(appsink);
        if (h264parse) gst_object_unref(h264parse);
        gst_object_unref(pipeline);
        return false;
    }

    GstPad *parse_src_pad = gst_element_get_static_pad(h264parse, "src");
    if (parse_src_pad) {
        gst_pad_add_probe(parse_src_pad, GST_PAD_PROBE_TYPE_BUFFER, rtspH264Probe, nullptr, nullptr);
        gst_object_unref(parse_src_pad);
    }
    else {
        Logger(WARN, "RTSP pipeline: could not attach H264 SEI probe");
    }

    gst_object_unref(h264parse);

    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        Logger(ERROR, "RTSP pipeline could not transition to PLAYING");
        gst_object_unref(appsink);
        gst_object_unref(pipeline);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(rtsp_gst_mutex);
        rtsp_pipeline = pipeline;
        rtsp_appsink = appsink;
    }

    Logger(INFO, "RTSP video receiver started: %s", url.c_str());
    return true;
}

static bool checkRtspBus() {
    GstElement *pipeline = nullptr;

    {
        std::lock_guard<std::mutex> lock(rtsp_gst_mutex);
        pipeline = rtsp_pipeline;
        if (pipeline) gst_object_ref(pipeline);
    }

    if (!pipeline) return false;

    GstBus *bus = gst_element_get_bus(pipeline);
    if (!bus) {
        gst_object_unref(pipeline);
        return true;
    }

    bool ok = true;
    GstMessage *msg = nullptr;

    while ((msg = gst_bus_pop(bus)) != nullptr) {
        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError *err = nullptr;
                gchar *dbg = nullptr;
                gst_message_parse_error(msg, &err, &dbg);
                Logger(ERROR, "RTSP/GStreamer error: %s", err ? err->message : "unknown");
                if (dbg) Logger(DEBUG, "RTSP/GStreamer debug: %s", dbg);
                if (err) g_error_free(err);
                if (dbg) g_free(dbg);
                ok = false;
            } break;

            case GST_MESSAGE_EOS:
                Logger(WARN, "RTSP/GStreamer EOS");
                ok = false;
                break;

            default:
                break;
        }

        gst_message_unref(msg);
    }

    gst_object_unref(bus);
    gst_object_unref(pipeline);

    return ok;
}

static bool pullRtspFrame(cv::Mat& frame, double& frame_ts_ms) {
    if (source_stop_requested.load()) return false;

    GstElement *appsink = nullptr;

    {
        std::lock_guard<std::mutex> lock(rtsp_gst_mutex);
        appsink = rtsp_appsink;
        if (appsink) gst_object_ref(appsink);
    }

    if (!appsink) return false;

    GstSample *sample = gst_app_sink_try_pull_sample(GST_APP_SINK(appsink), 1000000);
    gst_object_unref(appsink);

    if (!sample) return false;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstCaps *caps = gst_sample_get_caps(sample);

    if (!buffer || !caps) {
        gst_sample_unref(sample);
        return false;
    }

    GstStructure *s = gst_caps_get_structure(caps, 0);
    int width = 0;
    int height = 0;

    if (!gst_structure_get_int(s, "width", &width) || !gst_structure_get_int(s, "height", &height)) {
        gst_sample_unref(sample);
        return false;
    }

    RtspVideoStamp stamp;
    if (!takeRtspVideoStamp(GST_BUFFER_PTS(buffer), stamp)) {
        gst_sample_unref(sample);
        return false;
    }

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return false;
    }

    cv::Mat view(cv::Size(width, height), CV_8UC3, const_cast<guint8*>(map.data), cv::Mat::AUTO_STEP);
    frame = view.clone();

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    frame_ts_ms = normRtspTsUs(stamp.camera_ts_us);
    return true;
}

static bool pollRealSense(double * init_ts) {
    static std::deque<std::pair<double, vec3>> acc_buf;
    static std::deque<std::pair<double, vec3>> gyr_buf;
    static double last_frame_ts = -1.0;

    const size_t MAX_IMU_BUF = 500;
    const double EPS_MS = 1e-6;
    bool produced_packet = false;

    rs2::frame fframe;

    while (queue.poll_for_frame(&fframe)) {
        if (source_stop_requested.load() || runtimeIsPaused()) break;

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

        rs2::frameset aligned = fs;
        if (g_align_to_color != nullptr) {
            try {
                aligned = g_align_to_color->process(fs);
            }
            catch (const rs2::error& e) {
                Logger(WARN, "RealSense align to color failed: %s", e.what());
                continue;
            }
        }

        rs2::video_frame vf = aligned.get_color_frame();
        if (!vf) continue;

        rs2::depth_frame df = aligned.get_depth_frame();
        double frame_ts = normTs(vf.get_timestamp());

        if (last_frame_ts < 0.0) {
            last_frame_ts = frame_ts;
            continue;
        }

        if (source_config.cam.fps > 0.0) {
            const double min_frame_dt = 1000.0 / source_config.cam.fps;
            if ((frame_ts - last_frame_ts) + EPS_MS < min_frame_dt) continue;
        }

        SourceIn packet;
        packet.frame_dtms = frame_ts - last_frame_ts;
        packet.frame_tsms = frame_ts;
        if (source_config.gen.imu_on) {
            if (acc_buf.size() < 2 || gyr_buf.size() < 2) continue;
            if (frame_ts > acc_buf.back().first || frame_ts > gyr_buf.back().first) continue;

            std::vector<double> target_ts;

            for (const auto& g : gyr_buf) {
                if (g.first > last_frame_ts + EPS_MS && g.first < frame_ts - EPS_MS) target_ts.emplace_back(g.first);
            }

            if (target_ts.empty() || std::abs(target_ts.back() - frame_ts) > EPS_MS) target_ts.emplace_back(frame_ts);

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
                s.dt = (t - prev_t);
                s.vgyr = gyr_i;
                s.vacc = acc_i;

                packet.imu.emplace_back(s);

                prev_t = t;
            }

            if (packet.imu.empty()) continue;
        }

        cv::Mat out;
        const rs2_format color_fmt = vf.get_profile().format();

        if (color_fmt == RS2_FORMAT_RGB8) {
            cv::Mat img_rgb(cv::Size(vf.get_width(), vf.get_height()), CV_8UC3, (void*)vf.get_data(), cv::Mat::AUTO_STEP);
            cv::cvtColor(img_rgb, out, cv::COLOR_RGB2BGR);
        }
        else if (color_fmt == RS2_FORMAT_BGR8) {
            cv::Mat img_bgr(cv::Size(vf.get_width(), vf.get_height()), CV_8UC3, (void*)vf.get_data(), cv::Mat::AUTO_STEP);
            out = img_bgr;
        }
        else if (color_fmt == RS2_FORMAT_RGBA8) {
            cv::Mat img_rgba(cv::Size(vf.get_width(), vf.get_height()), CV_8UC4, (void*)vf.get_data(), cv::Mat::AUTO_STEP);
            cv::cvtColor(img_rgba, out, cv::COLOR_RGBA2BGR);
        }
        else if (color_fmt == RS2_FORMAT_BGRA8) {
            cv::Mat img_bgra(cv::Size(vf.get_width(), vf.get_height()), CV_8UC4, (void*)vf.get_data(), cv::Mat::AUTO_STEP);
            cv::cvtColor(img_bgra, out, cv::COLOR_BGRA2BGR);
        }
        else {
            Logger(WARN, "Unsupported color format: %s", rs2_format_to_string(color_fmt));
            continue;
        }

        packet.frame = out.clone();

        if (df) {
            const rs2_format depth_fmt = df.get_profile().format();
            packet.depth_tsms = normTs(df.get_timestamp());

            if (depth_fmt == RS2_FORMAT_Z16 || depth_fmt == RS2_FORMAT_Y16) {
                cv::Mat depth_view(cv::Size(df.get_width(), df.get_height()), CV_16UC1, const_cast<void*>(df.get_data()), cv::Mat::AUTO_STEP);
                cv::Mat depth_m;
                depth_view.convertTo(depth_m, CV_32F, df.get_units());
                packet.depth = depth_m.clone();
            }
            else if (depth_fmt == RS2_FORMAT_DISPARITY32) {
                cv::Mat depth_view(cv::Size(df.get_width(), df.get_height()), CV_32FC1, const_cast<void*>(df.get_data()), cv::Mat::AUTO_STEP);
                packet.depth = depth_view.clone();
            }
            else {
                packet.depth.release();
                packet.depth_tsms = 0.0;
                Logger(WARN, "Unsupported aligned depth format: %s", rs2_format_to_string(depth_fmt));
            }
        }
        else {
            packet.depth.release();
            packet.depth_tsms = 0.0;
        }

        if (source_config.gen.imu_on) {
            trimImuBuffer(acc_buf, frame_ts);
            trimImuBuffer(gyr_buf, frame_ts);
        }

        {
            std::lock_guard<std::mutex> lock(dout_mutex);
            vout.emplace_back(packet);
            while (vout.size() > MAX_HIST) vout.pop_front();
        }

        produced_packet = true;
        last_frame_ts = frame_ts;
    }

    (void)init_ts;
    return produced_packet;
}

static bool pollRtsp() {
    if (source_stop_requested.load()) return false;

    static double last_frame_ts = -1.0;

    const double EPS_MS = 1e-6;

    if (!rtspPipelineAvailable()) {
        if (!initRtspPipeline(source_config.gen.input)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return false;
        }
    }

    if (!checkRtspBus()) {
        closeRtspPipeline();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return false;
    }

    cv::Mat frame;
    double frame_ts = 0.0;

    if (!pullRtspFrame(frame, frame_ts)) return false;

    if (last_frame_ts < 0.0) {
        last_frame_ts = frame_ts;
        return false;
    }

    if (source_config.cam.fps > 0.0) {
        const double min_frame_dt = 1000.0 / source_config.cam.fps;
        if ((frame_ts - last_frame_ts) + EPS_MS < min_frame_dt) return false;
    }

    std::deque<std::pair<double, vec3>> acc_buf;
    std::deque<std::pair<double, vec3>> gyr_buf;

    {
        std::lock_guard<std::mutex> lock(rtsp_imu_mutex);
        acc_buf = rtsp_acc_buf;
        gyr_buf = rtsp_gyr_buf;
    }

    SourceIn packet;
    packet.frame_dtms = frame_ts - last_frame_ts;
    packet.frame_tsms = frame_ts;
    packet.frame = frame;
    packet.depth.release();
    packet.depth_tsms = 0.0;
    if (source_config.gen.imu_on) {
        if (acc_buf.size() < 2 || gyr_buf.size() < 2) return false;
        if (frame_ts > acc_buf.back().first || frame_ts > gyr_buf.back().first) return false;

        std::vector<double> target_ts;

        for (const auto& g : gyr_buf) {
            if (g.first > last_frame_ts + EPS_MS && g.first < frame_ts - EPS_MS) target_ts.emplace_back(g.first);
        }

        if (target_ts.empty() || std::abs(target_ts.back() - frame_ts) > EPS_MS) target_ts.emplace_back(frame_ts);

        double prev_t = last_frame_ts;

        for (double t : target_ts) {
            vec3 acc_i;
            vec3 gyr_i;

            const bool ok_acc = interpolateImu(acc_buf, t, acc_i);
            const bool ok_gyr = interpolateImu(gyr_buf, t, gyr_i);

            if (!ok_acc || !ok_gyr) continue;
            if (t <= prev_t + EPS_MS) continue;

            ImuSample s;
            s.ts = t;
            s.dt = t - prev_t;
            s.vacc = acc_i;
            s.vgyr = gyr_i;

            packet.imu.emplace_back(s);
            prev_t = t;
        }

        if (packet.imu.empty()) return false;

        {
            std::lock_guard<std::mutex> lock(rtsp_imu_mutex);
            trimImuBuffer(rtsp_acc_buf, frame_ts);
            trimImuBuffer(rtsp_gyr_buf, frame_ts);
        }
    }

    {
        std::lock_guard<std::mutex> lock(dout_mutex);
        vout.emplace_back(packet);
        while (vout.size() > MAX_HIST) vout.pop_front();
    }

    last_frame_ts = frame_ts;
    return true;
}

static std::string trimAscii(const std::string& value) {
    const size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";

    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

static std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, ',')) {
        tokens.emplace_back(trimAscii(token));
    }

    return tokens;
}

static int findHeaderIndex(const std::vector<std::string>& header, std::initializer_list<const char*> names) {
    for (size_t i = 0; i < header.size(); ++i) {
        const std::string key = lowerAscii(trimAscii(header[i]));
        for (const char* name : names) {
            if (key == name) return static_cast<int>(i);
        }
    }
    return -1;
}

static bool loadCsvImuSamples(const std::string& path, Config * config) {
    std::ifstream file(path);
    if (!file.is_open()) {
        Logger(ERROR, "initSource2: CSV file could not be opened: %s", path.c_str());
        return false;
    }

    csv_imu_samples.clear();
    csv_next_sample = 0;

    std::string line;
    if (!std::getline(file, line)) {
        Logger(ERROR, "initSource2: CSV file is empty: %s", path.c_str());
        return false;
    }

    const std::vector<std::string> header = splitCsvLine(line);
    const int ts_idx = findHeaderIndex(header, {"ts", "time", "timestamp", "t"});
    const int gx_idx = findHeaderIndex(header, {"gx", "wx", "gyrox", "gyrx"});
    const int gy_idx = findHeaderIndex(header, {"gy", "wy", "gyroy", "gyry"});
    const int gz_idx = findHeaderIndex(header, {"gz", "wz", "gyroz", "gyrz"});
    const int ax_idx = findHeaderIndex(header, {"ax", "accx"});
    const int ay_idx = findHeaderIndex(header, {"ay", "accy"});
    const int az_idx = findHeaderIndex(header, {"az", "accz"});

    if (ts_idx < 0 || gx_idx < 0 || gy_idx < 0 || gz_idx < 0 || ax_idx < 0 || ay_idx < 0 || az_idx < 0) {
        Logger(ERROR, "initSource2: CSV header must contain ts,gx,gy,gz,ax,ay,az");
        return false;
    }

    double last_ts_ms = 0.0;
    bool have_last_ts = false;
    double dt_sum_ms = 0.0;
    size_t dt_count = 0;

    while (std::getline(file, line)) {
        const std::string clean = trimAscii(line);
        if (clean.empty()) continue;

        const std::vector<std::string> cols = splitCsvLine(clean);
        const int max_idx = std::max({ts_idx, gx_idx, gy_idx, gz_idx, ax_idx, ay_idx, az_idx});
        if (static_cast<int>(cols.size()) <= max_idx) continue;

        try {
            const double ts_ms = std::stod(cols[ts_idx]);
            const double gx = std::stod(cols[gx_idx]);
            const double gy = std::stod(cols[gy_idx]);
            const double gz = std::stod(cols[gz_idx]);
            const double ax = std::stod(cols[ax_idx]);
            const double ay = std::stod(cols[ay_idx]);
            const double az = std::stod(cols[az_idx]);

            ImuSample sample;
            sample.ts = ts_ms;
            sample.dt = 0.0;
            if (have_last_ts) {
                sample.dt = ts_ms - last_ts_ms;
                if (sample.dt <= 0.0) continue;
                dt_sum_ms += sample.dt;
                ++dt_count;
            }

            sample.vgyr = vec3(gx, gy, gz);
            sample.vacc = vec3(ax, ay, az);
            csv_imu_samples.emplace_back(sample);

            last_ts_ms = ts_ms;
            have_last_ts = true;
        }
        catch (const std::exception&) {
            continue;
        }
    }

    if (csv_imu_samples.empty()) {
        Logger(ERROR, "initSource2: no valid IMU rows were found in CSV: %s", path.c_str());
        return false;
    }

    if (dt_count > 0 && config != nullptr) {
        const double mean_dt_ms = dt_sum_ms / static_cast<double>(dt_count);
        if (mean_dt_ms > 1e-9) config->imu.fps = 1000.0 / mean_dt_ms;
    }

    Logger(INFO,
           "initSource2: SOURCE_CSV loaded %zu IMU samples from %s at %.3f Hz",
           csv_imu_samples.size(),
           path.c_str(),
           config ? config->imu.fps : 0.0);
    return true;
}

static bool pollCsv() {
    {
        std::lock_guard<std::mutex> lock(dout_mutex);
        if (vout.size() >= MAX_HIST / 2) return false;
    }

    if (csv_next_sample >= csv_imu_samples.size()) {
        source_finished.store(true);
        return false;
    }

    SourceIn packet;
    packet.imu.emplace_back(csv_imu_samples[csv_next_sample++]);
    packet.frame_tsms = packet.imu.back().ts;
    packet.frame_dtms = packet.imu.back().dt;
    packet.depth.release();
    packet.depth_tsms = 0.0;

    {
        std::lock_guard<std::mutex> lock(dout_mutex);
        vout.emplace_back(packet);
        while (vout.size() > MAX_HIST) vout.pop_front();
    }

    if (csv_next_sample >= csv_imu_samples.size()) source_finished.store(true);
    return true;
}

bool initSource2(Config * config) {
    closeSource2();

    source_config = *config;
    source_finished.store(false);
    source_stop_requested.store(false);
    csv_imu_samples.clear();
    csv_next_sample = 0;

    {
        std::lock_guard<std::mutex> lock(dout_mutex);
        vout.clear();
    }

    rs2::config bag_config;

    if (source_config.gen.type == SOURCE_BAG) {
        bag_config.enable_device_from_file(source_config.gen.input, false);
        bag_config.enable_stream(RS2_STREAM_ACCEL);
        bag_config.enable_stream(RS2_STREAM_GYRO);

        if (source_config.gen.color_on) {
            bag_config.enable_stream(RS2_STREAM_COLOR);
            bag_config.enable_stream(RS2_STREAM_DEPTH);
            Logger(INFO, "Color feed enabled");
            Logger(INFO, "Depth feed enabled and aligned to color");
        }

        rs2::pipeline_profile profile = pipe.start(bag_config, [&](const rs2::frame& f) { queue.enqueue(f); });

        try {
            rs2::device bag_device = profile.get_device();
            if (bag_device.is<rs2::playback>()) {
                g_bag_playback = std::make_unique<rs2::playback>(bag_device.as<rs2::playback>());
            }
        }
        catch (const rs2::error&) {
            g_bag_playback.reset();
        }

        if (source_config.gen.color_on) {
            if (g_align_to_color != nullptr) delete g_align_to_color;
            g_align_to_color = new rs2::align(RS2_STREAM_COLOR);
        }

        float accfps = 0.0f;
        float gyrfps = 0.0f;
        for (const rs2::stream_profile& sp : profile.get_streams()) {
            const rs2_stream st = sp.stream_type();

            if (st == RS2_STREAM_COLOR) {
                rs2::video_stream_profile vsp = sp.as<rs2::video_stream_profile>();

                if (vsp) {
                    source_config.cam.width = static_cast<uint32_t>(vsp.width());
                    source_config.cam.height = static_cast<uint32_t>(vsp.height());
                    config->cam.width = source_config.cam.width;
                    config->cam.height = source_config.cam.height;

                    if (source_config.cam.fps > static_cast<double>(vsp.fps())) source_config.cam.fps = static_cast<double>(vsp.fps());
                    config->cam.fps = source_config.cam.fps;

                    const rs2_format fmt = vsp.format();
                    if (fmt == RS2_FORMAT_RGB8 || fmt == RS2_FORMAT_RGBA8 || fmt == RS2_FORMAT_BGR8 || fmt == RS2_FORMAT_BGRA8) {
                        // pollRealSense always outputs packet.frame in BGR order.
                        source_config.cam.is_rgb = false;
                    }
                    config->cam.is_rgb = source_config.cam.is_rgb;

                    Logger(INFO,
                           "initSource2: color profile width=%u height=%u bag_fps=%d cfg_fps=%.3f output_is_rgb=%d",
                           source_config.cam.width,
                           source_config.cam.height,
                           vsp.fps(),
                           config->cam.fps,
                           config->cam.is_rgb ? 1 : 0);
                }
            }
            else if (st == RS2_STREAM_DEPTH) {
                rs2::video_stream_profile vsp = sp.as<rs2::video_stream_profile>();
                if (vsp) Logger(INFO, "Depth profile: %dx%d fps=%d format=%s", vsp.width(), vsp.height(), vsp.fps(), rs2_format_to_string(vsp.format()));
            }else if (st == RS2_STREAM_ACCEL) {
                rs2::motion_stream_profile msp = sp.as<rs2::motion_stream_profile>();

                if (msp) {
                    accfps = static_cast<float>(msp.fps());
                }else {
                    accfps = static_cast<float>(sp.fps());
                }
            }
            else if (st == RS2_STREAM_GYRO) {
                rs2::motion_stream_profile msp = sp.as<rs2::motion_stream_profile>();

                if (msp) {
                    gyrfps = static_cast<float>(msp.fps());
                } else {
                    gyrfps = static_cast<float>(sp.fps());
                }
            }
        }
        config->imu.fps = gyrfps > accfps ?  gyrfps : accfps;
        Logger(INFO, "initSource2: imu fps set to %.3f", config->imu.fps);
    }
    else if (source_config.gen.type == SOURCE_RTSP) {
        setupBundledGStreamerRuntime();
        gst_init(nullptr, nullptr);

        source_config.gen.depth_on = false;
        config->gen.depth_on = false;
        source_config.cam.is_rgb = false;
        config->cam.is_rgb = false;

        const std::string host = hostFromRtspUrl(source_config.gen.input);

        if (!initRtspUdp(host)) {
            Logger(ERROR, "initSource2: RTSP UDP IMU receiver could not be started");
            return false;
        }

        if (!initRtspPipeline(source_config.gen.input)) {
            closeRtspUdp();
            Logger(ERROR, "initSource2: RTSP video receiver could not be started");
            return false;
        }

        Logger(INFO, "initSource2: SOURCE_RTSP enabled. video=%s imu_udp=%s:%u", source_config.gen.input.c_str(), host.c_str(), UDP_SEND_PORT);
    }
    else if (source_config.gen.type == SOURCE_PORT) {
        std::wstring widestr(source_config.gen.input.begin(), source_config.gen.input.end());

        if (!COMInit(husb, widestr.c_str())) {
            Logger(ERROR, "initSourceMan => Source %s could not be opened", source_config.gen.input.c_str());
            return false;
        }
    }
    else if (source_config.gen.type == SOURCE_CSV) {
        if (!loadCsvImuSamples(source_config.gen.input, config)) {
            return false;
        }
    }
    else {
        Logger(ERROR, "Unreachable: Source not available on init");
        return false;
    }

    source_open = true;
    poll_thread = std::thread(updateSource2, &source_config);
    return true;
}

static void updateSource2(Config * config) {
    static double init_ts = 0.0;

    while (!source_stop_requested.load()) {
        if (runtimeIsPaused()) {
            if (g_bag_playback) {
                try {
                    g_bag_playback->pause();
                }
                catch (const rs2::error&) {
                }
            }

            if (runtimeWaitIfPaused()) break;

            if (g_bag_playback) {
                try {
                    g_bag_playback->resume();
                }
                catch (const rs2::error&) {
                }
            }
        }
        if (source_stop_requested.load()) break;

        bool did_work = false;
        if (config->gen.type == SOURCE_BAG) did_work = pollRealSense(&init_ts);
        else if (config->gen.type == SOURCE_RTSP) did_work = pollRtsp();
        else if (config->gen.type == SOURCE_CSV) {
            did_work = pollCsv();
            if (!did_work && source_finished.load()) break;
        }
        else if (config->gen.type == SOURCE_PORT) {}
        else {
            Logger(ERROR, "Unreachable: Source not available while polling");
            break;
        }

        if (!did_work) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int getSource2(SourceIn * out) {
    if (!out) return -1;

    if (source_config.gen.type == SOURCE_BAG || source_config.gen.type == SOURCE_RTSP || source_config.gen.type == SOURCE_CSV) {
        std::lock_guard<std::mutex> lock(dout_mutex);
        if (vout.empty()) {
            if (source_config.gen.type == SOURCE_CSV && source_finished.load()) {
                *out = SourceIn{};
                return -1;
            }
            *out = SourceIn{};
            return 0;
        }
        *out = vout.front();
        vout.pop_front();
        return 1;
    }

    *out = SourceIn{};
    return 0;
}

void closeSource2() {
    if (!source_open && !poll_thread.joinable()) {
        source_stop_requested.store(false);
        source_finished.store(false);
        csv_imu_samples.clear();
        csv_next_sample = 0;
        return;
    }

    source_stop_requested.store(true);
    runtimeSetPaused(false);

    if (poll_thread.joinable()) {
        poll_thread.join();
    }

    if (source_config.gen.type == SOURCE_BAG) {
        try {
            pipe.stop();
        }
        catch (const rs2::error&) {
        }

        if (g_align_to_color != nullptr) {
            delete g_align_to_color;
            g_align_to_color = nullptr;
        }

        g_bag_playback.reset();
    }
    else if (source_config.gen.type == SOURCE_RTSP) {
        closeRtspPipeline();
        closeRtspUdp();
    }
    else if (source_config.gen.type == SOURCE_PORT && husb != INVALID_HANDLE_VALUE) {
        COMDeInit(husb);
        husb = INVALID_HANDLE_VALUE;
    }

    {
        std::lock_guard<std::mutex> lock(dout_mutex);
        vout.clear();
    }

    source_finished.store(false);
    csv_imu_samples.clear();
    csv_next_sample = 0;
    source_open = false;
}
