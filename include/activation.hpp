#pragma once
#include <cmath>

namespace xneur
{
    float sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

}