#pragma once
#include "config.hpp"

void imuPreInit(Config * config);
void imuPreUpdate(SourceIn * source, StateOut * state);
bool ResampleAccToGyroInPlace(SourceIn* src);