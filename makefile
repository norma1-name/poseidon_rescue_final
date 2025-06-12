CUDA_ARCH := -gencode arch=compute_75,code=sm_75   # works on T4; add others if needed
CPPFLAGS  := -std=c++17 -O3 -Iinclude -Xcompiler -Wall
NVFLAGS   := $(CUDA_ARCH) -O3 --use_fast_math -rdc=true

CPPFLAGS  := -std=c++17 -O3 -Iinclude -Xcompiler -Wall -Xcompiler -Wno-unknown-pragmas

TARGET := poseidon_rescue
SRC    := src/main.cu src/kernels.cu src/hash_constants_def.cu
OBJ    := $(SRC:.cu=.o)

%.o: %.cu
	@nvcc $(NVFLAGS) $(CPPFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	@nvcc $(NVFLAGS) $(CPPFLAGS) $^ -o $@ -lcudadevrt

clean:
	@rm -f $(OBJ) $(TARGET)

.PHONY: clean
