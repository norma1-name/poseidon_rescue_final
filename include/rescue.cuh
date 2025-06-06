#pragma once
/* ------------  Rescue permutation (t = 3)  over Goldilocks prime ------------ */

#include "field64.cuh"      // add_mod / mul_mod helpers

/* Constants live in GPU constant memory (filled from host) */
__constant__ uint64_t RESCUE_RC[64];    // round constants
__constant__ uint64_t RESCUE_M[9];      // 3 × 3 MDS (row-major)

/* ---- S-box exponents ------------------------------------------------------ */
constexpr uint64_t ALPHA      = 7;                      // odd rounds
constexpr uint64_t ALPHA_INV  = 0x92492491b6db6db7ULL;  // 7⁻¹ mod (p-1)

/* ---- helper: x^7 ---------------------------------------------------------- */
__device__ __forceinline__
uint64_t pow7(uint64_t x)
{
    uint64_t x2 = mul_mod(x, x);        // x²
    uint64_t x3 = mul_mod(x2, x);       // x³
    uint64_t x6 = mul_mod(x3, x3);      // x⁶
    return       mul_mod(x6, x);        // x⁷
}

/* ---- helper: x^ALPHA_INV  (un-rolled ladder, 12 multiplies) -------------- */
__device__ __forceinline__
uint64_t pow7inv(uint64_t x)
{
    /* ladder construction for e = 0x92492491B6DB6DB7 */
    uint64_t x2  = mul_mod(x, x);              // x²
    uint64_t x3  = mul_mod(x2, x);             // x³
    uint64_t x6  = mul_mod(x3, x3);            // x⁶
    uint64_t x7  = mul_mod(x6, x);             // x⁷
    uint64_t x14 = mul_mod(x7, x7);            // x¹⁴
    uint64_t x28 = mul_mod(x14, x14);          // x²⁸
    uint64_t x56 = mul_mod(x28, x28);          // x⁵⁶
    uint64_t x63 = mul_mod(x56, x7);           // x⁶³
    uint64_t x127= mul_mod(x63, x63);          // x¹²⁶
    uint64_t x255= mul_mod(x127, x127);        // x²⁵⁴
    return mul_mod( mul_mod(x255, x14),        // x²⁵⁴•x¹⁴
                    mul_mod(x3, x2) );         // x³•x²   → e = 0x9249...
}

/* -------------------------- Rescue permutation ---------------------------- */
template<int N_ROUNDS = 64>
__device__ void rescue_permute(uint64_t state[3])
{
    #pragma unroll
    for (int r = 0; r < N_ROUNDS; ++r)
    {
        /* ARK */
        state[0] = add_mod(state[0], RESCUE_RC[r]);

        /* S-box layer */
        if (r & 1) {                               // odd  → α = 7
            state[0] = pow7(state[0]);
            state[1] = pow7(state[1]);
            state[2] = pow7(state[2]);
        } else {                                   // even → α⁻¹
            state[0] = pow7inv(state[0]);
            state[1] = pow7inv(state[1]);
            state[2] = pow7inv(state[2]);
        }

        /* Linear MDS */
        uint64_t y0 = add_mod( mul_mod(RESCUE_M[0], state[0]),
                       add_mod( mul_mod(RESCUE_M[1], state[1]),
                                mul_mod(RESCUE_M[2], state[2])) );

        uint64_t y1 = add_mod( mul_mod(RESCUE_M[3], state[0]),
                       add_mod( mul_mod(RESCUE_M[4], state[1]),
                                mul_mod(RESCUE_M[5], state[2])) );

        uint64_t y2 = add_mod( mul_mod(RESCUE_M[6], state[0]),
                       add_mod( mul_mod(RESCUE_M[7], state[1]),
                                mul_mod(RESCUE_M[8], state[2])) );

        state[0] = y0;  state[1] = y1;  state[2] = y2;
    }
}
