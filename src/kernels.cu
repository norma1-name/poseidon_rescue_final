#include "poseidon.cuh"
#include "rescue.cuh"

extern "C" __global__
void poseidon_kernel(uint64_t* st,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) poseidon_permute(&st[i*3]);
}

extern "C" __global__
void rescue_kernel(uint64_t* st,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) rescue_permute(&st[i*3]);
}

/* (L,R) → parent  */
extern "C" __global__
void leaf_hash_kernel(const uint64_t* __restrict__ in,
                      uint64_t*      __restrict__ out,
                      int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    uint64_t s[3]={0ULL,in[2*i],in[2*i+1]};
    poseidon_permute(s);
    out[i]=s[0];
}

/* parent level */
extern "C" __global__
void merkle_level_kernel(const uint64_t* __restrict__ in,
                         uint64_t*      __restrict__ out,
                         int pairs){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=pairs) return;
    uint64_t s[3]={0ULL,in[2*i],in[2*i+1]};
    poseidon_permute(s);
    out[i]=s[0];
}
