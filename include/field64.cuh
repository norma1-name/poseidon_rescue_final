#pragma once
#include <stdint.h>

/* HD = works on both host & device */
#define HD __host__ __device__ __forceinline__

static constexpr uint64_t P = 0xffffffff00000001ULL;   // Goldilocks prime

/* modular add, mul (schoolbook is fine for demo) */
HD uint64_t add_mod(uint64_t a, uint64_t b) {
    uint64_t r = a + b;
    return (r >= P) ? r - P : r;
}
HD uint64_t mul_mod(uint64_t a, uint64_t b) {
    __uint128_t t = (__uint128_t)a * b;
    uint64_t lo = (uint64_t)t;
    uint64_t hi = (uint64_t)(t >> 64);
    uint64_t res = hi * 0xffffffffUL - hi + lo;   // P-Montgomery trick
    if (res >= P) res -= P;
    return res;
}

/* tiny exponent helpers */
HD uint64_t pow5(uint64_t x) {                // x⁵
    uint64_t x2 = mul_mod(x,x);
    uint64_t x4 = mul_mod(x2,x2);
    return mul_mod(x4,x);
}
HD uint64_t pow7(uint64_t x) {                // x⁷
    uint64_t x2 = mul_mod(x,x);
    uint64_t x3 = mul_mod(x2,x);
    uint64_t x6 = mul_mod(x3,x3);
    return mul_mod(x6,x);
}
HD uint64_t pow7inv(uint64_t x) {             // 7⁻¹ ladder (see paper)
    uint64_t x2  = mul_mod(x,x);
    uint64_t x3  = mul_mod(x2,x);
    uint64_t x6  = mul_mod(x3,x3);
    uint64_t x7  = mul_mod(x6,x);
    uint64_t x14 = mul_mod(x7,x7);
    uint64_t x28 = mul_mod(x14,x14);
    uint64_t x56 = mul_mod(x28,x28);
    uint64_t x63 = mul_mod(x56,x7);
    uint64_t x127= mul_mod(x63,x63);
    uint64_t x255= mul_mod(x127,x127);
    return mul_mod(mul_mod(x255,x14), mul_mod(x3,x2));
}
