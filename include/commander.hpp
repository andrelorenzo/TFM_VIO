#pragma once

#include "config.hpp"

void commanderInit(Config * config);
void commanderSend(const Command& cmd);
void commanderClose();
