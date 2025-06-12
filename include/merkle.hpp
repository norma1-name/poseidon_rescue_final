#pragma once
#include <vector>
#include <cstdint>

struct MerkleProof {
    std::vector<uint64_t> siblings;
    uint32_t              leafIndex;
};

inline MerkleProof generate_proof(uint32_t idx,
                                  const std::vector<std::vector<uint64_t>>& L)
{
    MerkleProof p; p.leafIndex = idx;
    for (const auto& layer : L) {
        p.siblings.push_back(layer[idx ^ 1U]);
        idx >>= 1;
    }
    return p;
}

/* forward decl */
uint64_t poseidon_hash2(uint64_t, uint64_t);

inline bool verify_proof(uint64_t leafHash,
                         const MerkleProof& p,
                         uint64_t root)
{
    uint64_t h = leafHash;
    uint32_t idx = p.leafIndex;
    for (uint64_t sib : p.siblings) {
        h = (idx & 1) ? poseidon_hash2(sib,h)
                      : poseidon_hash2(h,sib);
        idx >>= 1;
    }
    return h == root;
}
