#pragma once

#include "config.hpp"

void controllerInit(const Config * config);
void controllerUpdate(StateOut& state, Command * cmd);
