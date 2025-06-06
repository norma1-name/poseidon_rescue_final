#include "poseidon.cuh"
#include "rescue.cuh"

/* ------------------------------------------------------------------ */
/* Poseidon permutation kernel: 1 thread = 1 state (3 × uint64)       */
/* ------------------------------------------------------------------ */
extern "C" __global__
void poseidon_kernel(uint64_t* states, int n_states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;
    poseidon_permute<64>(&states[idx * 3]);
}

/* ------------------------------------------------------------------ */
/* Rescue permutation kernel                                          */
/* ------------------------------------------------------------------ */
extern "C" __global__
void rescue_kernel(uint64_t* states, int n_states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;
    rescue_permute(&states[idx * 3]);
}

/* ------------------------------------------------------------------ */
/* Leaf-hash kernel: hashes two 64-bit blocks → 1 parent hash         */
/* Uses Poseidon; 24 bytes I/O per leaf                               */
/* ------------------------------------------------------------------ */
extern "C" __global__
void leaf_hash_kernel(const uint64_t* __restrict__ in,
                      uint64_t*       __restrict__ out,
                      int n_leaves)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_leaves) return;

    // Build state: [0, left, right]
    uint64_t st[3];
    st[0] = 0ULL;
    st[1] = in[2*i];       // left child
    st[2] = in[2*i + 1];   // right child

    // Permute with Poseidon
    poseidon_permute<64>(st);

    // Write parent = state[0]
    out[i] = st[0];
}

/* ------------------------------------------------------------------ */
/* Merkle-level kernel: hashes pairs of parent nodes → next-level     */
/* Also uses Poseidon; input and output arrays are consecutive hashes */
/* ------------------------------------------------------------------ */
extern "C" __global__
void merkle_level_kernel(const uint64_t* __restrict__ in,
                         uint64_t*       __restrict__ out,
                         int n_pairs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pairs) return;

    // Build state: [0, parent_left, parent_right]
    uint64_t st[3];
    st[0] = 0ULL;
    st[1] = in[2*i];
    st[2] = in[2*i + 1];

    // Permute with Poseidon
    poseidon_permute<64>(st);

    // Write grand-parent = state[0]
    out[i] = st[0];
}