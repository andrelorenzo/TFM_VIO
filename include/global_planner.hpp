#pragma once

#include "config.hpp"

void globalPlanInit(const Config * config);
void globalPlanUpdate(StateOut& state, Waypoints& path);
