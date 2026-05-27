#pragma once

#include "config.hpp"

void localPlannerInit(Config * config);
void localPlannerUpdate(const EvitationDir& dir, StateOut& state, const Waypoints& path, Command * cmd);
