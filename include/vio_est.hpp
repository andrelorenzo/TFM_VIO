#pragma once
#include "config.hpp"
bool vioUpdate(SourceIn * source, StateOut * state);
void vioInit(Config config);
void vioClose();

cv::Mat getDebugImage();