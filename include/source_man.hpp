#pragma once

#include "config.hpp"
#include <cstdint>
#include <deque>
#include <opencv2/opencv.hpp>

struct sourcePacket
{
    std::deque<imuData> imu_data;
    cv::Mat color;
    cv::Mat depth;              // CV_32F en metros, alineado con color

    bool new_color = false;
    bool new_depth = false;

    double colorts_ms = 0.0;
    double depthts_ms = 0.0;
    std::uint64_t color_frame_number = 0;
};

bool initSourceManager(Config& config);
void closeSourceManager();

bool getSourceManager(sourcePacket* packet);
bool getSourceManager(cv::Mat* frame, imuData* imu);
