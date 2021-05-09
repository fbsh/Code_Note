#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "common/Error.h"
#include "common/GpuTimer.h"
#include "common/Vector.h"

#define N 500
#define FULL_DATA_SIZE 100000000
#define DIMGRID 10
#define DIMBLOCK 10
#define XMIN -10.0f
#define XMAX 10.0f

const int ARRAY_BYTES = N * sizeof(float);
const int FULL_ARRAY_BYTES = FULL_DATA_SIZE * sizeof(float);

__host__ __device__ float function1(float x) {
    return x * x;
}

__host__ __device__ float function2(float x) {
    return sinf(x);
}

__global__ void functionKernel1(Vector<float> d_a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float x, dx;

    dx = (XMAX - (XMIN)) / ((float)N - 1);
    while (i < n) {
        x = XMIN + i * dx;
        d_a.setElement(i, function1(x));
        i += blockDim.x * gridDim.x;
    }
}

__global__ void functionKernel2(Vector<float> d_a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float x, dx;

    dx = (XMAX - (XMIN)) / ((float)N - 1);

    while (i < n) {
        x = XMIN + i * dx;
        d_a.setElement(i, function2(x));
        i += blockDim.x * gridDim.x;
    }
}

void onDevice(Vector<float> h_a, Vector<float> h_b) {
    Vector<float> d_a, d_b;

    // create the stream
    cudaStream_t stream1;
    HANDLER_ERROR_ERR(cudaStreamCreate(&stream1));

    GpuTimer timer;
    timer.Start(stream1);

    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_a.elements, ARRAY_BYTES));
    HANDLER_ERROR_ERR(cudaMalloc((void**)&d_b.elements, ARRAY_BYTES));

    for (int i = 0; i < FULL_DATA_SIZE; i += N) {
        // copy the locked memory to the device, async
        HANDLER_ERROR_ERR(cudaMemcpyAsync(d_a.elements, h_a.elements + i,
                                          ARRAY_BYTES, cudaMemcpyHostToDevice,
                                          stream1));

        // copy the locked memory to the device, async
        HANDLER_ERROR_ERR(cudaMemcpyAsync(d_b.elements, h_b.elements + i,
                                          ARRAY_BYTES, cudaMemcpyHostToDevice,
                                          stream1));

        functionKernel1<<<DIMGRID, DIMBLOCK, 0, stream1>>>(d_a, N);
        HANDLER_ERROR_MSG("kernel panic!!!");

        functionKernel2<<<DIMGRID, DIMBLOCK, 0, stream1>>>(d_b, N);
        HANDLER_ERROR_MSG("kernel panic!!!");

        /*
        HANDLER_ERROR_ERR(cudaMemcpy(h_a.elements, d_a.elements, ARRAY_BYTES,
        cudaMemcpyDeviceToHost)); HANDLER_ERROR_ERR(cudaMemcpy(h_b.elements,
        d_b.elements, ARRAY_BYTES, cudaMemcpyDeviceToHost));
        */

        // copy from the device to the locked memory, async
        HANDLER_ERROR_ERR(cudaMemcpyAsync(h_a.elements + i, d_a.elements,
                                          ARRAY_BYTES, cudaMemcpyDeviceToHost,
                                          stream1));

        HANDLER_ERROR_ERR(cudaMemcpyAsync(h_b.elements + i, d_b.elements,
                                          ARRAY_BYTES, cudaMemcpyDeviceToHost,
                                          stream1));
    }

    // synchronization
    HANDLER_ERROR_ERR(cudaStreamSynchronize(stream1));

    // stop timer
    timer.Stop(stream1);

    // print time
    printf("Time :  %f ms\n", timer.Elapsed());

    // destroy stream
    HANDLER_ERROR_ERR(cudaStreamDestroy(stream1));

    // free device memory
    HANDLER_ERROR_ERR(cudaFree(d_a.elements));
    HANDLER_ERROR_ERR(cudaFree(d_b.elements));
}

void checkDeviceProps() {
    cudaDeviceProp prop;
    int whichDevice;
    HANDLER_ERROR_ERR(cudaGetDevice(&whichDevice));
    HANDLER_ERROR_ERR(cudaGetDeviceProperties(&prop, whichDevice));
    if (!prop.deviceOverlap) {
        printf(
            "Device will not handle overlaps, so no speed up from streams\n");
    }
}

void test() {
    Vector<float> h_a, h_b;
    h_a.length = FULL_DATA_SIZE;
    h_b.length = FULL_DATA_SIZE;

    /* Not used because of the cudaHostAlloc
    h_a.elements = (float*)malloc( ARRAY_BYTES );
    h_b.elements = (float*)malloc( ARRAY_BYTES );
    */

    // allocate host locked memory
    HANDLER_ERROR_ERR(cudaHostAlloc((void**)&h_a.elements, FULL_ARRAY_BYTES,
                                    cudaHostAllocDefault));
    HANDLER_ERROR_ERR(cudaHostAlloc((void**)&h_b.elements, FULL_ARRAY_BYTES,
                                    cudaHostAllocDefault));

    int i;
    for (i = 0; i < FULL_DATA_SIZE; i++) {
        h_a.setElement(i, 1.0);
        h_b.setElement(i, 1.0);
    }

    // call device configuration
    onDevice(h_a, h_b);

    /* Not used because of the cudaHostAlloc
    free(h_a.elements);
    free(h_b.elements);
    */
    for (i = 0; i < FULL_DATA_SIZE; i++) {
        assert(h_a.getElement(i) != 1);
        assert(h_b.getElement(i) != 1);
    }

    printf("-: successful execution :-\n");

    // free host memory
    HANDLER_ERROR_ERR(cudaFreeHost(h_a.elements));
    HANDLER_ERROR_ERR(cudaFreeHost(h_b.elements));
}

int main() {
    checkDeviceProps();
    test();
}
