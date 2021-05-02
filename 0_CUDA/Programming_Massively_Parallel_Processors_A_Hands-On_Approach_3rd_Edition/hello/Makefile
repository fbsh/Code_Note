NVCC = /usr/local/cuda/bin/nvcc

all: helloworld

%.o : %.cu
	$(NVCC) -c $< -o $@

helloworld : helloworld.o
	$(NVCC) $^ -o $@

clean:
	rm -rf *.o *.a helloworld