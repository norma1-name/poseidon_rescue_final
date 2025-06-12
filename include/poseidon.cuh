#pragma once
#include "field64.cuh"
#include "hash_constants.cuh"
#include "poseidon_rc.hpp"
#include "poseidon_mds.hpp"

/* Select table pointers */
#ifdef __CUDA_ARCH__
    #define RC64(i) POSEIDON_RC[i]
    #define M9(i)  POSEIDON_M [i]
#else
    #define RC64(i) poseidon_rc[i]
    #define M9(i)  poseidon_mds[i]
#endif

HD void poseidon_permute(uint64_t s[3]) {
    #pragma unroll
    for (int r = 0; r < 64; ++r) {
        s[0] = add_mod(s[0], RC64(r));

        s[0] = pow5(s[0]);  s[1] = pow5(s[1]);  s[2] = pow5(s[2]);

        uint64_t y0 = add_mod(mul_mod(M9(0),s[0]),
                     add_mod(mul_mod(M9(1),s[1]),
                              mul_mod(M9(2),s[2])));
        uint64_t y1 = add_mod(mul_mod(M9(3),s[0]),
                     add_mod(mul_mod(M9(4),s[1]),
                              mul_mod(M9(5),s[2])));
        uint64_t y2 = add_mod(mul_mod(M9(6),s[0]),
                     add_mod(mul_mod(M9(7),s[1]),
                              mul_mod(M9(8),s[2])));
        s[0]=y0; s[1]=y1; s[2]=y2;
    }
}
