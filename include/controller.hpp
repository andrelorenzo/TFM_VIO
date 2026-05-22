#pragma once

#include "config.hpp"

void controllerInit(const Config * config);
void controllerUpdate(const StateOut& state, Command * cmd);
