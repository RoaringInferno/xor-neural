#pragma once

#include <cmath>

namespace xneur
{
    struct cost_function
    {
        float (*cost)(float output, float expected);
        float (*derivative)(float output, float expected);
    };

    namespace cost
    {
        const cost_function mean_squared_error = {
            [](float output, float expected) { return 0.5f * (output - expected) * (output - expected); },
            [](float output, float expected) { return output - expected; }
        };

        const cost_function cross_entropy = {
            [](float output, float expected) { return -expected * std::log(output) - (1.0f - expected) * std::log(1.0f - output); },
            [](float output, float expected) { return (output - expected) / (output * (1.0f - output)); }
        };
    }
}