#pragma once

#include "config.hpp"

void da3Init(const Config* config);
void da3Update(const SourceIn* source);
EvitationDir da3Get();
cv::Mat da3GetDebugImage();
void da3Close();
