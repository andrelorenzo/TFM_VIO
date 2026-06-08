#include "commander.hpp"
#include "config.hpp"
#include "seconds/comms_common.h"

#include <atomic>
#include <cmath>
#include <chrono>
#include <cstring>

static SOCKET hsocket = INVALID_SOCKET;

#define MSGID_TWIST_CMD 0x0021
#define COMMANDER_PORT UDP_CTRL_PORT

#pragma pack(push, 1)
typedef struct {
    uint32_t seq;
    uint64_t ts_us;
    float lin[3]; // m/s: x forward, y left, z up
    float ang[3]; // rad/s: roll, pitch, yaw
} twist_cmd_t;
#pragma pack(pop)

static std::atomic<uint32_t> cmd_seq{0};
static std::wstring commander_ip = UDP_IP;
static uint16_t commander_port = COMMANDER_PORT;

static void closeCommanderSocket(){
    if(hsocket != INVALID_SOCKET){
        closesocket(hsocket);
        hsocket = INVALID_SOCKET;
    }
}

static bool initCommanderSocket(){
    WSADATA wsdata;
    if(WSAStartup(MAKEWORD(2, 2), &wsdata) != 0){
        return false;
    }

    SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if(sock == INVALID_SOCKET){
        return false;
    }

    DWORD sndTimeoutMs = 50;
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&sndTimeoutMs), sizeof(sndTimeoutMs));

    hsocket = sock;
    return true;
}

static std::wstring widenAscii(const std::string& value){
    return std::wstring(value.begin(), value.end());
}

static bool isIpv4Literal(const std::wstring& value){
    sockaddr_in addr = {};
    return InetPtonW(AF_INET, value.c_str(), &addr.sin_addr) == 1;
}

static std::string hostFromRtspUrl(const std::string& url){
    const std::string prefix = "rtsp://";
    size_t begin = 0;

    if(url.compare(0, prefix.size(), prefix) == 0) begin = prefix.size();

    const size_t slash = url.find('/', begin);
    std::string authority = url.substr(begin, slash == std::string::npos ? std::string::npos : slash - begin);

    const size_t at = authority.find('@');
    if(at != std::string::npos) authority = authority.substr(at + 1);

    if(!authority.empty() && authority[0] == '['){
        const size_t end_bracket = authority.find(']');
        if(end_bracket != std::string::npos) return authority.substr(1, end_bracket - 1);
    }

    const size_t colon = authority.find(':');
    if(colon != std::string::npos) authority = authority.substr(0, colon);

    if(authority.empty()) authority = RTSP_IP;
    return authority;
}

static uint64_t nowUs(){
    auto now = std::chrono::steady_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return static_cast<uint64_t>(us);
}

static uint64_t commandTsUs(double ts_ms){
    if(std::isfinite(ts_ms) && ts_ms > 0.0) return static_cast<uint64_t>(std::llround(ts_ms * 1000.0));
    return nowUs();
}

static void configureCommander(Config * config){
    closeCommanderSocket();

    if(!initCommanderSocket()){
        hsocket = INVALID_SOCKET;
        return;
    }

    commander_ip = UDP_IP;
    commander_port = COMMANDER_PORT;

    if(config != nullptr && config->gen.type == SOURCE_RTSP){
        commander_ip = widenAscii(hostFromRtspUrl(config->gen.input));
    }

#ifdef UDP_CTRL_IP
    commander_ip = UDP_CTRL_IP;
#endif

#ifdef UDP_CTRL_PORT
    commander_port = UDP_CTRL_PORT;
#endif

    if(!isIpv4Literal(commander_ip)){
        commander_ip = UDP_IP;
    }
}

void commanderInit(Config * config){
    configureCommander(config);
}

void commanderSend(const Command& cmd){
    if(hsocket == INVALID_SOCKET) return;

    twist_cmd_t twist = {};
    twist.seq = cmd_seq.fetch_add(1) + 1;
    twist.ts_us = commandTsUs(cmd.ts_ms);

    twist.lin[0] = static_cast<float>(cmd.lenvel_ms.x());
    twist.lin[1] = static_cast<float>(cmd.lenvel_ms.y());
    twist.lin[2] = static_cast<float>(cmd.lenvel_ms.z());

    twist.ang[0] = static_cast<float>(cmd.angvel_rads.x());
    twist.ang[1] = static_cast<float>(cmd.angvel_rads.y());
    twist.ang[2] = static_cast<float>(cmd.angvel_rads.z());

    comms_payload_t payload = {};
    payload.msg_id = MSGID_TWIST_CMD;
    payload.data_size = sizeof(twist_cmd_t);
    memcpy(payload.data, &twist, sizeof(twist_cmd_t));

    Crc32 crc;
    payload.crc_32 = crc.Calculate(&payload);

    uint8_t frame[BUFFER_DEFAULT_SIZE + 64] = {};
    int frame_len = EncodeFrame(&payload, frame);

    if(frame_len <= 0){
        Logger(WARN, "Commander: EncodeFrame fallo msg_id=0x%04X payload_size=%u", payload.msg_id, payload.data_size);
        return;
    }


    const int sent = UDPSend(hsocket, commander_ip, commander_port, frame, static_cast<size_t>(frame_len));
    if(sent < 0){
        Logger(WARN, "Commander TX fallo -> %S:%u err=%d", commander_ip.c_str(), commander_port, sent);
    }
}

void commanderClose(){
    closeCommanderSocket();
}
