#pragma once

#include "config.hpp"

void globalPlanInit(const Config * config);
void globalPlanUpdate(const StateOut& state, Waypoints& path);
