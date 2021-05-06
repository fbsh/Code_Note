NVCC = /usr/local/cuda/bin/nvcc

all: Norm_CUDA

%.o : %.cu
	$(NVCC) -c $< -o $@

Norm_CUDA : Norm_CUDA.o
	$(NVCC) $^ -o $@

clean:
	rm -rf *.o *.a Norm_CUDA