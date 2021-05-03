#include <stdio.h>

__global__ void hello() {
    printf("Hello world from device\n");
}

int main() {
    hello<<<1, 10>>>();
    printf("Hello world from host\n");
    cudaDeviceSynchronize();
    return 0;
}
