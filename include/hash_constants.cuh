#pragma once
#include <stdint.h>

#ifdef DEFINE_HASH_CONSTANTS
__constant__ uint64_t POSEIDON_RC[64] = { /* … */ };
__constant__ uint64_t POSEIDON_M [ 9] = { /* … */ };
__constant__ uint64_t RESCUE_RC [64]  = { /* … */ };
__constant__ uint64_t RESCUE_M  [ 9]  = { /* … */ };
#else
extern __constant__ uint64_t POSEIDON_RC[];
extern __constant__ uint64_t POSEIDON_M[];
extern __constant__ uint64_t RESCUE_RC[];
extern __constant__ uint64_t RESCUE_M[];
#endif
