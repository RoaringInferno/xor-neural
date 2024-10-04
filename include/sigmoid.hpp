#pragma once

#include <math.h>

#define M_PI_2        1.57079632679489661923	/* pi/2 */
#define M_PI_2_INV    (1.0/M_PI_2)
#define M_2_SQRTPI    1.12837916709551257390    /* 2/sqrt(pi) */
#define ERF_COEF      (1.0/M_2_SQRTPI)

// #define SIGMOID_USE_ATAN
#define SIGMOID_USE_EXP
// #define SIGMOID_USE_SQRT
// #define SIGMOID_USE_ERF
// #define SIGMOID_USE_FABS

#ifdef SIGMOID_USE_ATAN
#define SIGMOID_LAMBDA [](float x) -> float { return M_PI_2_INV*atan(M_PI_2*x); }
#endif

#ifdef SIGMOID_USE_EXP
#define SIGMOID_LAMBDA [](float x) -> float { return 1.0/(1.0 + exp(-x)); }
#endif

#ifdef SIGMOID_USE_SQRT
#define SIGMOID_LAMBDA [](float x) -> float { return 1.0/sqrt(1.0 + x*x); }
#endif

#ifdef SIGMOID_USE_ERF
#define SIGMOID_LAMBDA [](float x) -> float { return erf(ERF_COEF*x); }
#endif

#ifdef SIGMOID_USE_FABS
#define SIGMOID_LAMBDA [](float x) -> float { return x/(1.0 + fabs(x)); }
#endif