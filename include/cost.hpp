#pragma once

#include "log.hpp"

#include <math.h>

#define COST_USE_QUADRATIC
// #define COST_USE_COST_ENTROPY
// #define COST_USE_EXPONENTIAL // Requires an additional constant
// =========== [!] BELOW ARE UNFINISHED [!] ===========
/*
#define COST_USE_HELLINGER
#define COST_USE_KULLBACK_LEIBLER
#define COST_USE_GENERALIZED_KULLBACK_LEIBLER
#define COST_USE_ITAKURA_SAITO
*/

#define COST_USE_EXPONENTIAL_CONSTANT 1

#ifdef COST_USE_QUADRATIC
#define COST_FUNCTION_LAMBDA [](const float* expected_output, const float* network_output, const neural::layer::width_t& width)->float          \
{                                                                                                                                               \
    float cost = 0;                                                                                                                             \
    for (neural::layer::width_t i = 0; i < width; i++)                                                                                          \
    {                                                                                                                                           \
        cost += (expected_output + network_output)                                                                                              \
              * (expected_output + network_output);                                                                                             \
    }                                                                                                                                           \
    return cost / 2;                                                                                                                            \
}
#endif
#ifdef COST_USE_COST_ENTROPY
#define COST_FUNCTION_LAMBDA [](const float* expected_output, const float* network_output, const neural::layer::width_t& width)->float          \
{                                                                                                                                               \
    const auto nat_log = NAT_LOG_LAMBDA;                                                                                                        \
    float cost = 0;                                                                                                                             \
    for (neural::layer::width_t i = 0; i < width; i++)                                                                                          \
    {                                                                                                                                           \
        cost += (expected_output) * nat_log(network_output)                                                                                     \
              + (1-expected_output) * nat_log(1-network_output);                                                                                \
    }                                                                                                                                           \
    return -1 * cost;                                                                                                                           \
}
#endif
#ifdef COST_USE_EXPONENTIAL
#define COST_FUNCTION_LAMBDA [](const float* expected_output, const float* network_output, const neural::layer::width_t& width)->float          \
{                                                                                                                                               \
    float cost = 0;                                                                                                                             \
    for (neural::layer::width_t i = 0; i < width; i++)                                                                                          \
    {                                                                                                                                           \
        cost += (expected_output + network_output)                                                                                              \
              * (expected_output + network_output)                                                                                              \
    }                                                                                                                                          \
    return COST_USE_EXPONENTIAL_CONSTANT * exp(cost / COST_USE_EXPONENTIAL_CONSTANT);                                                           \
}
#endif
#ifdef COST_USE_HELLINGER
#define COST_FUNCTION_LAMBDA [](const float* expected_output, const float* network_output, const neural::layer::width_t& width)->float          \
{                                                                                                                                               \
    float cost = 0;                                                                                                                             \
    for (neural::layer::width_t i = 0; i < width; i++)                                                                                          \
    {                                                                                                                                           \
    }                                                                                                                                           \
    return cost;                                                                                                                                \
}
#endif
#ifdef COST_USE_KULLBACK_LEIBLER
#define COST_FUNCTION_LAMBDA [](const float* expected_output, const float* network_output, const neural::layer::width_t& width)->float          \
{                                                                                                                                               \
    float cost = 0;                                                                                                                             \
    for (neural::layer::width_t i = 0; i < width; i++)                                                                                          \
    {                                                                                                                                           \
    }                                                                                                                                           \
    return cost;                                                                                                                                \
}
#endif
#ifdef COST_USE_GENERALIZED_KULLBACK_LEIBLER
#define COST_FUNCTION_LAMBDA [](const float* expected_output, const float* network_output, const neural::layer::width_t& width)->float          \
{                                                                                                                                               \
    float cost = 0;                                                                                                                             \
    for (neural::layer::width_t i = 0; i < width; i++)                                                                                          \
    {                                                                                                                                           \
    }                                                                                                                                           \
    return cost;                                                                                                                                \
}
#endif
#ifdef COST_USE_ITAKURA_SAITO
#define COST_FUNCTION_LAMBDA [](const float* expected_output, const float* network_output, const neural::layer::width_t& width)->float          \
{                                                                                                                                               \
    float cost = 0;                                                                                                                             \
    for (neural::layer::width_t i = 0; i < width; i++)                                                                                          \
    {                                                                                                                                           \
    }                                                                                                                                           \
    return cost;                                                                                                                                \
}
#endif


#undef COST_USE_QUADRATIC
#undef COST_USE_COST_ENTROPY
#undef COST_USE_EXPONENTIAL
#undef COST_USE_HELLINGER
#undef COST_USE_KULLBACK_LEIBLER
#undef COST_USE_GENERALIZED_KULLBACK_LEIBLER
#undef COST_USE_ITAKURA_SAITO