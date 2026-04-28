#pragma once
#include "config.hpp"

enum SourceManType{
    SOURCEMAN_OK,
    SOURCEMAN_ERR,
    SOURCEMAN_EOF
};

bool initSourceMan(Config * config);
void closeSourceMan();
SourceManType getSourceMan(SourceIn * out);