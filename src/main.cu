#include <vector>
#include <random>
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#include "poseidon_rc.hpp"
#include "poseidon_mds.hpp"
#include "rescue_rc.hpp"
#include "rescue_mds.hpp"

#include "field64.cuh"
#include "hash_constants.cuh"
#include "poseidon.cuh"
#include "rescue.cuh"
#include "merkle.hpp"

/* ── kernels ────────────────────────────────────────────────────── */
extern "C" __global__ void poseidon_kernel    (uint64_t*, int);
extern "C" __global__ void rescue_kernel      (uint64_t*, int);
extern "C" __global__ void leaf_hash_kernel   (const uint64_t*, uint64_t*, int);
extern "C" __global__ void merkle_level_kernel(const uint64_t*, uint64_t*, int);

/* ── CUDA error helper ──────────────────────────────────────────── */
#define CUDA_OK(call) gpuAssert((call), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t e,const char* f,int l){
    if(e!=cudaSuccess){
        fprintf(stderr,"CUDA error %s (%s:%d)\n",
                cudaGetErrorString(e),f,l); exit(EXIT_FAILURE);
    }
}

/* host copy of Poseidon tables */
struct PoseidonTables { uint64_t rc[64]; uint64_t m[9]; } gPoseidon;

/* host Poseidon 2→1 */
uint64_t poseidon_hash2(uint64_t a,uint64_t b)
{
    uint64_t s[3] = {0ULL,a,b};
    for(int r=0;r<64;++r){
        s[0] = add_mod(s[0], gPoseidon.rc[r]);
        s[0] = pow5(s[0]);  s[1] = pow5(s[1]);  s[2] = pow5(s[2]);
        uint64_t y0 = add_mod(mul_mod(gPoseidon.m[0],s[0]),
                     add_mod(mul_mod(gPoseidon.m[1],s[1]),
                              mul_mod(gPoseidon.m[2],s[2])));
        uint64_t y1 = add_mod(mul_mod(gPoseidon.m[3],s[0]),
                     add_mod(mul_mod(gPoseidon.m[4],s[1]),
                              mul_mod(gPoseidon.m[5],s[2])));
        uint64_t y2 = add_mod(mul_mod(gPoseidon.m[6],s[0]),
                     add_mod(mul_mod(gPoseidon.m[7],s[1]),
                              mul_mod(gPoseidon.m[8],s[2])));
        s[0]=y0; s[1]=y1; s[2]=y2;
    }
    return s[0];
}

/* copy host→device constant & print sizes */
template<typename T,size_t N>
static void copyConstant(const char* name,const T(&host)[N],const void* dev){
    size_t devSz; CUDA_OK(cudaGetSymbolSize(&devSz,dev));
    size_t n = (sizeof(host)<devSz)?sizeof(host):devSz;
    printf("%-10s host=%4zu  device=%4zu  → copying %zu bytes\n",
           name,sizeof(host),devSz,n);
    CUDA_OK(cudaMemcpyToSymbol(dev,host,n));
}

constexpr int BLOCK = 256;


int main()
{
    /* 0. constant ------------------------------------------ */
    copyConstant("POSEIDON_RC",poseidon_rc, POSEIDON_RC);
    copyConstant("POSEIDON_M", poseidon_mds,POSEIDON_M );
    copyConstant("RESCUE_RC",  rescue_rc,   RESCUE_RC );
    copyConstant("RESCUE_M",   rescue_mds,  RESCUE_M  );
    CUDA_OK(cudaMemcpyFromSymbol(gPoseidon.rc,POSEIDON_RC,sizeof(gPoseidon.rc)));
    CUDA_OK(cudaMemcpyFromSymbol(gPoseidon.m ,POSEIDON_M ,sizeof(gPoseidon.m )));

    /* 1. permutation benchmarks ----------------------------------- */
    const int BATCH = 1<<20;
    std::vector<uint64_t> hStates(3ull*BATCH);
    std::mt19937_64 rng(42); for(auto& w:hStates) w=rng();
    uint64_t* dStates; CUDA_OK(cudaMalloc(&dStates,hStates.size()*8));
    CUDA_OK(cudaMemcpy(dStates,hStates.data(),hStates.size()*8,
                       cudaMemcpyHostToDevice));
    dim3 gPerm((BATCH+BLOCK-1)/BLOCK);
    cudaEvent_t t0,t1; float ms;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    cudaEventRecord(t0); poseidon_kernel<<<gPerm,BLOCK>>>(dStates,BATCH);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1);
    printf("Poseidon throughput   : %.2f MH/s\n",BATCH/ms*1e3/1e6);

    cudaEventRecord(t0); rescue_kernel<<<gPerm,BLOCK>>>(dStates,BATCH);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms,t0,t1);
    printf("Rescue   throughput   : %.2f MH/s\n",BATCH/ms*1e3/1e6);
    cudaFree(dStates);

    /* 2. leaf hashing --------------------------------------------- */
    const int LEAVES = 1<<22;                    // 4 194 304 leaves
    uint64_t *leaf_in_d,*leaf_out_d;
    CUDA_OK(cudaMalloc(&leaf_in_d,2ull*LEAVES*8));
    CUDA_OK(cudaMalloc(&leaf_out_d,      LEAVES*8));
    std::vector<uint64_t> tmp(2ull*LEAVES);
    for(auto& w:tmp) w=rng();
    CUDA_OK(cudaMemcpy(leaf_in_d,tmp.data(),tmp.size()*8,
                       cudaMemcpyHostToDevice));

    dim3 gLeaf((LEAVES+BLOCK-1)/BLOCK);
    leaf_hash_kernel<<<gLeaf,BLOCK>>>(leaf_in_d,leaf_out_d,LEAVES);
    CUDA_OK(cudaDeviceSynchronize());            // warm-up

    cudaEventRecord(t0);
    leaf_hash_kernel<<<gLeaf,BLOCK>>>(leaf_in_d,leaf_out_d,LEAVES);
    cudaEventRecord(t1); CUDA_OK(cudaEventSynchronize(t1));
    cudaEventElapsedTime(&ms,t0,t1);
    double GBps=(LEAVES*24.0)/(ms*1e-3)/(1u<<30);
    printf("Leaf-hash throughput  : %.2f GB/s (%.0f ms for %d leaves)\n",
           GBps,ms,LEAVES);

    /* --- diagnostic: compare leaf-0 CPU vs GPU ------------------- */
    uint64_t child[2]; CUDA_OK(cudaMemcpy(child,leaf_in_d,16,cudaMemcpyDeviceToHost));
    uint64_t gpuLeaf0; CUDA_OK(cudaMemcpy(&gpuLeaf0,leaf_out_d,8,cudaMemcpyDeviceToHost));
    uint64_t cpuLeaf0 = poseidon_hash2(child[0],child[1]);
    printf("DEBUG leaf-0          : GPU %016llx | CPU %016llx %s\n",
           (unsigned long long)gpuLeaf0,
           (unsigned long long)cpuLeaf0,
           gpuLeaf0==cpuLeaf0?"MATCH":"MISMATCH");

    /* 3. copy fresh leaf hashes & build Merkle tree ---------------- */
    std::vector<std::vector<uint64_t>> levels;
    levels.emplace_back(LEAVES);
    CUDA_OK(cudaMemcpy(levels.back().data(),leaf_out_d,LEAVES*8,
                       cudaMemcpyDeviceToHost));

    uint64_t* in_d=leaf_out_d; uint64_t* buf_d; CUDA_OK(cudaMalloc(&buf_d,LEAVES*8));
    int nodes=LEAVES, depth=0;
    while(nodes>1){
        int pairs=nodes>>1;
        dim3 gLvl((pairs+BLOCK-1)/BLOCK);
        merkle_level_kernel<<<gLvl,BLOCK>>>(in_d,buf_d,pairs);
        CUDA_OK(cudaDeviceSynchronize());
        levels.emplace_back(pairs);
        CUDA_OK(cudaMemcpy(levels.back().data(),buf_d,pairs*8,
                           cudaMemcpyDeviceToHost));
        std::swap(in_d,buf_d); nodes=pairs; ++depth;
    }
    uint64_t root = levels.back()[0];   
    levels.pop_back();                  
    printf("Merkle root (level %2d): %016llx\n", depth,
          (unsigned long long)root);

    /* --- diagnostic: level-1 node-0 CPU vs GPU ------------------- */
    uint64_t cpuL1 = poseidon_hash2(levels[0][0], levels[0][1]);
    printf("DEBUG level-1         : GPU %016llx | CPU %016llx %s\n",
           (unsigned long long)levels[1][0],
           (unsigned long long)cpuL1,
           cpuL1==levels[1][0]?"MATCH":"MISMATCH");

    /* 4. proof ----------------------------------------------------- */
    uint32_t idx = 123456;
    uint64_t leafHash = levels[0][idx];
    MerkleProof proof = generate_proof(idx,levels);
    bool ok = verify_proof(leafHash,proof,root);
    printf("Proof for leaf %u      : %s\n",idx, ok?"VALID":"INVALID");

    /* cleanup */
    cudaFree(buf_d); cudaFree(leaf_out_d); cudaFree(leaf_in_d);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return 0;
}