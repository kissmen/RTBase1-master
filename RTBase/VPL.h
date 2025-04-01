#pragma once
#include "Core.h"
#include "Materials.h"

class VPL {
public:
    ShadingData shadingData;
    Colour flux;

    VPL(const ShadingData& _shadingData, const Colour& _flux) : shadingData(_shadingData), flux(_flux) {}
};
