//  src/poseidon_consts.cu
#include "poseidon.cuh"
#include "rescue.cuh"

// ONE AND ONLY definition of the device-side tables
__constant__ uint64_t POSEIDON_RC[48];
__constant__ uint64_t POSEIDON_M [ 9];
__constant__ uint64_t RESCUE_RC [48];
__constant__ uint64_t RESCUE_M  [ 9];
