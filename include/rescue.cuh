#pragma once
#include "field64.cuh"
#include "hash_constants.cuh"
#include "rescue_rc.hpp"
#include "rescue_mds.hpp"

#ifdef __CUDA_ARCH__
    #define RRC64(i) RESCUE_RC[i]
    #define RM9(i)   RESCUE_M [i]
#else
    #define RRC64(i) rescue_rc[i]
    #define RM9(i)   rescue_mds[i]
#endif

HD void rescue_permute(uint64_t s[3]) {
    #pragma unroll
    for (int r = 0; r < 64; ++r) {
        s[0] = add_mod(s[0], RRC64(r));

        if (r & 1) { s[0]=pow7(s[0]);  s[1]=pow7(s[1]);  s[2]=pow7(s[2]); }
        else       { s[0]=pow7inv(s[0]); s[1]=pow7inv(s[1]); s[2]=pow7inv(s[2]); }

        uint64_t y0 = add_mod(mul_mod(RM9(0),s[0]),
                     add_mod(mul_mod(RM9(1),s[1]),
                              mul_mod(RM9(2),s[2])));
        uint64_t y1 = add_mod(mul_mod(RM9(3),s[0]),
                     add_mod(mul_mod(RM9(4),s[1]),
                              mul_mod(RM9(5),s[2])));
        uint64_t y2 = add_mod(mul_mod(RM9(6),s[0]),
                     add_mod(mul_mod(RM9(7),s[1]),
                              mul_mod(RM9(8),s[2])));
        s[0]=y0; s[1]=y1; s[2]=y2;
    }
}
