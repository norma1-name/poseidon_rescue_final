#pragma once
#include "field64.cuh"

/* ---- round constants & MDS matrix in constant memory ---- */
__constant__ uint64_t POSEIDON_RC[64];   // adjust to spec
__constant__ uint64_t POSEIDON_M[9];     // 3×3 matrix, row-major

template<int N_ROUNDS>
__device__ void poseidon_permute(uint64_t state[3])
{
    #pragma unroll
    for (int r = 0; r < N_ROUNDS; ++r)
    {
        state[0] = add_mod(state[0], POSEIDON_RC[r]);   // ARK

        /* S-box (full round, power 5) */
        state[0] = pow5(state[0]);
        state[1] = pow5(state[1]);
        state[2] = pow5(state[2]);

        /* MDS mix */
        uint64_t y0 = add_mod(mul_mod(POSEIDON_M[0], state[0]),
                    add_mod(mul_mod(POSEIDON_M[1], state[1]),
                             mul_mod(POSEIDON_M[2], state[2])));
        uint64_t y1 = add_mod(mul_mod(POSEIDON_M[3], state[0]),
                    add_mod(mul_mod(POSEIDON_M[4], state[1]),
                             mul_mod(POSEIDON_M[5], state[2])));
        uint64_t y2 = add_mod(mul_mod(POSEIDON_M[6], state[0]),
                    add_mod(mul_mod(POSEIDON_M[7], state[1]),
                             mul_mod(POSEIDON_M[8], state[2])));
        state[0] = y0;  state[1] = y1;  state[2] = y2;
    }
}
