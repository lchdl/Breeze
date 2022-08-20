#pragma once
#include "basedefs.h"
#include "basemath.h"
#include "nn/nn_global.h"

/* From wikipedia: https://en.wikipedia.org/wiki/Xorshift */
/* xorwow is used as default PRNG in CUDA Toolkit. */
/* The state array must be initialized to not be all zero in the first four words */
struct xorwow_state {
    unsigned int x[5];
    unsigned int counter;
};
unsigned int xorwow(xorwow_state * state);

class XorwowRNG {
protected:
    xorwow_state state;
public:
    XorwowRNG();
    XorwowRNG(const int& a, const int& b);
    void init();
    void init(const int& a, const int& b);
    REAL uniform();
    REAL uniform(const REAL& min, const REAL& max);
    REAL gaussian();
    REAL gaussian(const REAL& mu, const REAL& sigma);
};

/* generate normally distributed random variable using xorwow method */
REAL uniform_distribution();
REAL uniform_distribution(REAL min, REAL max);

/* generate normally distributed random variable using the Box-Muller method */
REAL normal_distribution();
REAL normal_distribution(REAL mean, REAL std);

