#include <vector>
#include <random>
#include <cstdio>
#include <cuda_runtime.h>
#include "rescue_rc.hpp"
#include "rescue_mds.hpp"

CUDA_OK( cudaMemcpyToSymbol(RESCUE_RC, rescue_rc,  sizeof(rescue_rc)) );
CUDA_OK( cudaMemcpyToSymbol(RESCUE_M,  rescue_mds, sizeof(rescue_mds)) );

extern void poseidon_kernel(uint64_t*, int);   // forward decl

/* helper: check CUDA errs */
#define CUDA_OK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* f, int l)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA err %s %s:%d\n",
                cudaGetErrorString(code), f, l); exit(code);
    }
}

int main() {
    const int BATCH = 1<<20;                  // 1 048 576 states
    std::vector<uint64_t> h(3*BATCH);
    std::mt19937_64 rng(42); for (auto& x: h) x = rng();

    uint64_t* d; CUDA_OK(cudaMalloc(&d, h.size()*8));
    CUDA_OK(cudaMemcpy(d, h.data(), h.size()*8, cudaMemcpyHostToDevice));

    dim3 threads(256), blocks((BATCH+255)/256);

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    poseidon_kernel<<<blocks,threads>>>(d, BATCH);
    cudaEventRecord(t1); cudaEventSynchronize(t1);

    float ms; cudaEventElapsedTime(&ms,t0,t1);
    printf("Poseidon %d-state batch: %.2f MH/s\n",
           BATCH, BATCH/ms*1e3/1e6);

    cudaEventRecord(t0);
    rescue_kernel<<<blocks,threads>>>(d, BATCH);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1);
    printf("Rescue   %d-state batch: %.2f MH/s\n", BATCH, BATCH/ms*1e3/1e6);

    cudaFree(d);
    return 0;
}
