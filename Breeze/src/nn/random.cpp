#include "nn/random.h"

XorwowRNG _NN_rng;

unsigned int simple_hash(unsigned int a)
{
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}
unsigned int xorwow(xorwow_state * state)
{
    /* Algorithm "xorwow" from p. 5 of Marsaglia, "Xorshift RNGs" */
    unsigned int t = state->x[4];

    unsigned int s = state->x[0];  /* Perform a contrived 32-bit shift. */
    state->x[4] = state->x[3];
    state->x[3] = state->x[2];
    state->x[2] = state->x[1];
    state->x[1] = s;

    t ^= t >> 2;
    t ^= t << 1;
    t ^= s ^ (s << 4);
    state->x[0] = t;
    state->counter += 362437;
    return t + state->counter;
}

XorwowRNG::XorwowRNG() { init(); }
XorwowRNG::XorwowRNG(const int & a, const int & b)
{
    init(a, b);
}
void XorwowRNG::init() { init(13579, 24680); }
void XorwowRNG::init(const int & a, const int & b)
{
    /* init global state using two independent parameters a and b. */
    state.x[0] = simple_hash((a ^ 0x7123bbcc) + (b ^ 0x0baabfcb));
    state.x[1] = simple_hash((a ^ 0xfabbcddc) + 7 + (b ^ 0xa30fb67a));
    state.x[2] = simple_hash((a ^ 0x0078ddcc) - 23 + (b ^ 0xffaabccb));
    state.x[3] = simple_hash((a ^ 0x78633ff0) + 1001 + (b ^ 0x98ab47f1));
    state.x[4] = simple_hash((a ^ 0xfe910725) - 7301 + (b ^ 0xacfe9712));
    state.counter = simple_hash((a ^ 0x0893ff87) - (b ^ 0x19d86b2d));
}

REAL XorwowRNG::uniform()
{
    unsigned int r = xorwow(&state);
    return (REAL(r) / REAL(0xffffffff));
}
REAL XorwowRNG::uniform(const REAL & min, const REAL & max) { return uniform() * (max - min) + min; }
REAL XorwowRNG::gaussian()
{
    REAL  u, r, theta;           /* Variables for Box-Muller method */
    REAL  x;                     /* Normal(0, 1) rv */

    /* Generate u */
    u = REAL(0.0);
    while (u == REAL(0.0))
        u = uniform();

    /* Compute r */
    r = Sqrt(REAL(-2.0) * Log(u));

    /* Generate theta */
    theta = REAL(0.0);
    while (theta == REAL(0.0))
        theta = REAL(2.0) * PI * uniform();

    /* Generate x value */
    x = r * Cos(theta);

    /* Return the normally distributed RV value */
    return x;
}
REAL XorwowRNG::gaussian(const REAL & mu, const REAL & sigma){return (gaussian() * sigma) + mu;}
REAL uniform_distribution() { return _NN_rng.uniform(); }
REAL uniform_distribution(REAL min, REAL max) { return _NN_rng.uniform(min, max); }
REAL normal_distribution() { return _NN_rng.gaussian(); }
REAL normal_distribution(REAL mean, REAL std) { return _NN_rng.gaussian(mean, std); }
