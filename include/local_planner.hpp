#pragma once

#include "config.hpp"

void localPlannerInit(Config * config);
void localPlannerUpdate(EvitationDir dir, StateOut state, Waypoints path, Command * cmd);
