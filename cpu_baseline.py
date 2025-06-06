import re, time, random, math, pathlib

# ---- 1. Load round constants & MDS from the C++ headers -----------------
def load_hex_list(path):
    txt = pathlib.Path(path).read_text()
    return [int(x, 16) for x in re.findall(r'0x[0-9a-fA-F]+', txt)]

P = (1 << 64) - (1 << 32) + 1

poseidon_rc = load_hex_list('include/poseidon_rc.hpp')
poseidon_m  = load_hex_list('include/poseidon_mds.hpp')
rescue_rc   = load_hex_list('include/rescue_rc.hpp')
rescue_m    = load_hex_list('include/rescue_mds.hpp')

# ---- 2. Field helpers ----------------------------------------------------
def add(x, y): return (x + y) % P
def mul(x, y): return (x * y) % P
def pow5(x):
    x2 = mul(x, x)
    x4 = mul(x2, x2)
    return mul(x4, x)

ALPHA_INV = 0x92492491b6db6db7  # 7⁻¹ mod (P-1)
def pow7(x):
    return pow(x, 7, P)
def pow7inv(x):
    return pow(x, ALPHA_INV, P)

# ---- 3. Poseidon permutation (t = 3, 64 rounds) -------------------------
def poseidon_perm(state):
    s0, s1, s2 = state
    rc_iter = iter(poseidon_rc)
    for _ in range(64):
        s0 = add(s0, next(rc_iter))        # ARK
        s0, s1, s2 = map(pow5, (s0, s1, s2))
        # MDS mul
        y0 = add(mul(poseidon_m[0], s0),
                 add(mul(poseidon_m[1], s1), mul(poseidon_m[2], s2)))
        y1 = add(mul(poseidon_m[3], s0),
                 add(mul(poseidon_m[4], s1), mul(poseidon_m[5], s2)))
        y2 = add(mul(poseidon_m[6], s0),
                 add(mul(poseidon_m[7], s1), mul(poseidon_m[8], s2)))
        s0, s1, s2 = y0, y1, y2
    return (s0, s1, s2)

# ---- 4. Rescue permutation ----------------------------------------------
def rescue_perm(state):
    s0, s1, s2 = state
    rc_iter = iter(rescue_rc)
    for r in range(64):
        s0 = add(s0, next(rc_iter))
        if r & 1:
            s0, s1, s2 = map(pow7, (s0, s1, s2))
        else:
            s0, s1, s2 = map(pow7inv, (s0, s1, s2))
        y0 = add(mul(rescue_m[0], s0),
                 add(mul(rescue_m[1], s1), mul(rescue_m[2], s2)))
        y1 = add(mul(rescue_m[3], s0),
                 add(mul(rescue_m[4], s1), mul(rescue_m[5], s2)))
        y2 = add(mul(rescue_m[6], s0),
                 add(mul(rescue_m[7], s1), mul(rescue_m[8], s2)))
        s0, s1, s2 = y0, y1, y2
    return (s0, s1, s2)

# ---- 5. Benchmark --------------------------------------------------------
BATCH = 100_000   # 1e5 gives stable numbers in <1 s

rand64 = lambda: random.randrange(P)
states = [(0, rand64(), rand64()) for _ in range(BATCH)]

t = time.perf_counter()
[poseidon_perm(list(s)) for s in states]
pose_ms = (time.perf_counter() - t) * 1e3

t = time.perf_counter()
[rescue_perm(list(s)) for s in states]
resc_ms = (time.perf_counter() - t) * 1e3

print(f"Poseidon CPU: {BATCH/pose_ms:.2f} MH/s  ({pose_ms:.0f} ms)")
print(f"Rescue  CPU: {BATCH/resc_ms:.2f} MH/s  ({resc_ms:.0f} ms)")
