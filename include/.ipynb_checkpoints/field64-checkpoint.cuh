#pragma once
#include <cstdint>

/* Prime p = 2^64 − 2^32 + 1  (fits in uint64_t) */
constexpr uint64_t P      = 0xffffffff00000001ULL;
constexpr uint64_t P_INV  = 0xffffffff00000000ULL;   // −p⁻¹ mod 2^64 (Barrett)


__device__ __forceinline__
uint64_t add_mod(uint64_t a, uint64_t b) {
    uint64_t c = a + b;
    return c - P * (c >= P);                 
}

__device__ __forceinline__
uint64_t mul_mod(uint64_t a, uint64_t b) {
    __uint128_t t = static_cast<__uint128_t>(a) * b;   // 128-bit product
    uint64_t q = static_cast<uint64_t>(t >> 64) * P_INV;
    uint64_t r = static_cast<uint64_t>(t) - static_cast<__uint128_t>(q) * P;
    return r >= P ? r - P : r;
}

/* x ↦ x^5 */
__device__ __forceinline__
uint64_t pow5(uint64_t x) {
    uint64_t x2 = mul_mod(x, x);
    uint64_t x4 = mul_mod(x2, x2);
    return mul_mod(x4, x);
}
