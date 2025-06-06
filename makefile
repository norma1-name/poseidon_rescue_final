CUDA_ARCH := -gencode arch=compute_61,code=sm_61 
             
CPPFLAGS   := -std=c++17 -O3 -Iinclude -Xcompiler -Wall

NVFLAGS    := $(CUDA_ARCH) -O3 --use_fast_math

TARGET     := poseidon_rescue
SRC        := src/main.cu src/kernels.cu

$(TARGET): $(SRC)
	@nvcc $(NVFLAGS) $(CPPFLAGS) $(SRC) -o $(TARGET)

clean:
	@rm -f $(TARGET)

.PHONY: clean