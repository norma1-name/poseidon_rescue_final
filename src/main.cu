#include <vector>
#include <random>
#include <cstdio>
#include <cuda_runtime.h>

#include "poseidon_rc.hpp"
#include "poseidon_mds.hpp"
#include "rescue_rc.hpp"
#include "rescue_mds.hpp"

#include "poseidon.cuh"
#include "rescue.cuh"

/* -------- kernel prototypes (defined in kernels.cu) ---------------- */
extern "C" __global__ void poseidon_kernel(uint64_t*, int);
extern "C" __global__ void rescue_kernel (uint64_t*, int);
extern "C" __global__ void leaf_hash_kernel   (const uint64_t*, uint64_t*, int);
extern "C" __global__ void merkle_level_kernel(const uint64_t*, uint64_t*, int);

/* -------- CUDA-error helper ---------------------------------------- */
#define CUDA_OK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,const char* f,int l){
    if(code!=cudaSuccess){
        fprintf(stderr,"CUDA %s %s:%d\n",
                cudaGetErrorString(code),f,l);
        exit(code);
    }
}

/* ================================================================== */
int main()
{
    /* 0. copy constants to __constant__ memory ---------------------- */
    CUDA_OK(cudaMemcpyToSymbol(POSEIDON_RC, poseidon_rc, sizeof(poseidon_rc)));
    CUDA_OK(cudaMemcpyToSymbol(POSEIDON_M , poseidon_mds,sizeof(poseidon_mds)));
    CUDA_OK(cudaMemcpyToSymbol(RESCUE_RC  , rescue_rc  , sizeof(rescue_rc)));
    CUDA_OK(cudaMemcpyToSymbol(RESCUE_M   , rescue_mds , sizeof(rescue_mds)));

    dim3 t256(256);
    cudaEvent_t t0, t1; 
    float ms; 
    cudaEventCreate(&t0); 
    cudaEventCreate(&t1);

    /* ================================================================
     * 1. permutation timing (1 048 576 states)
     * ============================================================== */
    const int BATCH = 1<<20;
    std::vector<uint64_t> host(3 * BATCH);
    std::mt19937_64 rng(42);
    for (auto& x : host) x = rng();

    uint64_t* d; 
    CUDA_OK(cudaMalloc(&d, host.size() * sizeof(uint64_t)));
    CUDA_OK(cudaMemcpy(d, host.data(), host.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    dim3 bPerm((BATCH + 255) / 256);
    poseidon_kernel<<<bPerm, t256>>>(d, BATCH);   // warm-up
    rescue_kernel<<<bPerm, t256>>>(d, BATCH);
    CUDA_OK(cudaDeviceSynchronize());

    /* Poseidon timed */
    cudaEventRecord(t0);
    poseidon_kernel<<<bPerm, t256>>>(d, BATCH);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    printf("Poseidon %d-state batch : %.2f MH/s\n",
           BATCH, BATCH / ms * 1e3 / 1e6);

    /* Rescue timed */
    cudaEventRecord(t0);
    rescue_kernel<<<bPerm, t256>>>(d, BATCH);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    printf("Rescue   %d-state batch : %.2f MH/s\n",
           BATCH, BATCH / ms * 1e3 / 1e6);

    /* ================================================================
     * 2. leaf-hash bandwidth (random leaves → parents)
     * ============================================================== */
    const int LEAVES = 1<<22;                             // 4,194,304 leaves
    uint64_t *leaf_in_d, *leaf_out_d;
    CUDA_OK(cudaMalloc(&leaf_in_d,  LEAVES * 2ull * sizeof(uint64_t)));  // L,R (16 B)
    CUDA_OK(cudaMalloc(&leaf_out_d, LEAVES * sizeof(uint64_t)));         // parent (8 B)

    /* Fill leaf_in_d on host then copy to device */
    std::vector<uint64_t> tmp(LEAVES * 2);
    std::mt19937_64 prng(123);
    for (auto& x : tmp) x = prng();
    CUDA_OK(cudaMemcpy(leaf_in_d,
                       tmp.data(),
                       tmp.size() * sizeof(uint64_t),
                       cudaMemcpyHostToDevice));

    /* Debug: peek first few leaf inputs on GPU */
    {
        uint64_t debugL[4] = {0,0,0,0};
        CUDA_OK(cudaMemcpy(debugL, leaf_in_d, sizeof(debugL), cudaMemcpyDeviceToHost));
        printf("debug leaf_in_d[0..3]: %016llx %016llx %016llx %016llx\n",
               (unsigned long long)debugL[0],
               (unsigned long long)debugL[1],
               (unsigned long long)debugL[2],
               (unsigned long long)debugL[3]);
    }

    dim3 bLeaf((LEAVES + 255) / 256);
    leaf_hash_kernel<<<bLeaf, t256>>>(leaf_in_d, leaf_out_d, LEAVES); // warm-up
    CUDA_OK(cudaDeviceSynchronize());

    /* Debug: peek first few leaf outputs (parents) */
    {
        uint64_t debugP[4] = {0,0,0,0};
        CUDA_OK(cudaMemcpy(debugP, leaf_out_d, sizeof(debugP), cudaMemcpyDeviceToHost));
        printf("debug leaf_out_d[0..3]: %016llx %016llx %016llx %016llx\n",
               (unsigned long long)debugP[0],
               (unsigned long long)debugP[1],
               (unsigned long long)debugP[2],
               (unsigned long long)debugP[3]);
    }

    cudaEventRecord(t0);
    leaf_hash_kernel<<<bLeaf, t256>>>(leaf_in_d, leaf_out_d, LEAVES);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    double GBps = (LEAVES * 24.0) / (ms * 1e-3) / (1<<30);   // 24 B per leaf
    printf("Leaf-hash throughput  : %.2f GB/s  (%.0f ms for %d leaves)\n",
           GBps, ms, LEAVES);

    /* ================================================================
     * 3. Merkle-tree build (leaf → root)
     * ============================================================== */
    uint64_t *in_d  = leaf_out_d;                      // level-0 parents
    uint64_t *buf_d; 
    CUDA_OK(cudaMalloc(&buf_d, LEAVES * sizeof(uint64_t)));

    int nodes = LEAVES;
    int level = 0;
    while (nodes > 1) {
        int pairs = nodes >> 1;
        dim3 bLvl((pairs + 255) / 256);

        /* Debug: peek first parents before running level 1 */
        if (level == 0) {
            uint64_t debugL0[4] = {0,0,0,0};
            CUDA_OK(cudaMemcpy(debugL0, in_d, sizeof(debugL0), cudaMemcpyDeviceToHost));
            printf("debug level0 parents[0..3]: %016llx %016llx %016llx %016llx\n",
                   (unsigned long long)debugL0[0],
                   (unsigned long long)debugL0[1],
                   (unsigned long long)debugL0[2],
                   (unsigned long long)debugL0[3]);
        }

        merkle_level_kernel<<<bLvl, t256>>>(in_d, buf_d, pairs);
        std::swap(in_d, buf_d);
        nodes = pairs;
        ++level;
    }
    CUDA_OK(cudaDeviceSynchronize());

    uint64_t root;
    CUDA_OK(cudaMemcpy(&root, in_d, sizeof(root), cudaMemcpyDeviceToHost));
    printf("Merkle root after %d levels: %016llx\n",
           level, (unsigned long long)root);

    /* ---- cleanup --------------------------------------------------- */
    cudaFree(buf_d);
    cudaFree(leaf_in_d);
    cudaFree(leaf_out_d);
    cudaFree(d);
    return 0;
}
