#include "Error.h"
#include "GpuTimer.h"
#include "Vector.h"

#define N 16
#define BLOCK_SIZE 2
#define STRIDE 4
#define POW(x) (x) * (x)

__global__ void mseKernel(Vector<float> d_a,
                          Vector<float> d_b,
                          Vector<float> d_c) {
    // -:YOUR CODE HERE:-
}

void onDevice(Vector<float> h_a, Vector<float> h_b, Vector<float> h_c) {
    // declare GPU vectors
    // -:YOUR CODE HERE:-

    // start timer
    GpuTimer timer;
    timer.Start();

    const int ARRAY_BYTES = N * sizeof(float);

    // allocate  memory on the GPU
    // -:YOUR CODE HERE:-

    // copy data from CPU the GPU
    // -:YOUR CODE HERE:-

    // execution configuration
    dim3 GridBlocks(2, 2);
    dim3 ThreadsBlocks(2, 2);

    mseKernel<<<GridBlocks, ThreadsBlocks>>>(d_a, d_b, d_c);
    HANDLER_ERROR_MSG("kernel panic!!!");

    // copy data back from the GPU to the CPU
    // -:YOUR CODE HERE:-

    // stop timer
    timer.Stop();

    // print time
    printf("Time :  %f ms\n", timer.Elapsed());

    // free GPU memory
    // -:YOUR CODE HERE:-
}

void test() {
    Vector<float> h_a, h_b, h_c;
    h_a.length = N;
    h_b.length = N;
    h_c.length = 4;

    h_a.elements = (float*)malloc(h_a.length * sizeof(float));
    h_b.elements = (float*)malloc(h_a.length * sizeof(float));
    h_c.elements = (float*)malloc(4 * sizeof(float));

    int i, j = 16, k = 1;

    for (i = 0; i < h_a.length; i++) {
        h_a.setElement(i, k);
        h_b.setElement(i, j);
        // h_c.setElement( i, 0.0 );
        k++;
        j--;
    }

    // call device configuration
    onDevice(h_a, h_b, h_c);

    // verify that the GPU did the work we requested
    for (int i = 0; i < 4; i++) {
        printf(" [%i] = %f \n", i, h_c.getElement(i));
    }

    printf("-: successful execution :-\n");

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements);
}

int main(void) {
    test();
    return 0;
}
