/* Matrix normalization.
 * Compile with "gcc matrixNorm.c"
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define CHECK_ERR(x)                            \
if (x != cudaSuccess) {                         \
fprintf(stderr,"%s in %s at line %d\n",         \
cudaGetErrorString(err),__FILE__,__LINE__);     \
exit(-1);                                       \
}                                               \


/* Program Parameters */
#define N 6000  /* Matrix size */

/* Matrices */
volatile float A[N][N], B[N][N];

__global__ void vecAdd (float* d_A, float* d_B, int n) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (col<n){

    int row;
    float mu, sigma; // Mean and Standard Deviation
    mu = 0.0;
    for (row=0; row < n; row++){
            mu += d_A[row*N+col];
    }
    mu /= (float) n;
    sigma = 0.0;
    for (row=0; row < n; row++){
        sigma += powf(d_A[row*N+col] - mu, 2.0);
    }
    sigma /= (float) n;
    sigma = sqrt(sigma);
    for (row=0; row < n; row++) {
        if (sigma == 0.0)
            d_B[row*N+col] = 0.0;
        else
            d_B[row*N+col] = (d_A[row*N+col] - mu) / sigma;
    }

  }

}

/* Initialize A and B*/
void initialize_inputs()   {
    int row, col;

    srand((unsigned)time(NULL));
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[row][col] = (float)rand() / 32768.0;
            B[row][col] = 0.0;
        }
    }

}


/* Kernel function */

void matrixNorm() {

    cudaError_t err;
    //start from here:
    float *d_A, *d_B;
    err = cudaMalloc((void **) &d_A, sizeof(float)*N*N);
       CHECK_ERR(err);
    err = cudaMalloc((void **) &d_B, sizeof(float)*N*N);
       CHECK_ERR(err);
    err = cudaMemcpy(d_A, (void*)A , sizeof(float)*N*N, cudaMemcpyHostToDevice);
        CHECK_ERR(err);

    //kernal function here
    vecAdd<<<ceil(N/256.0), 256>>>(d_A,d_B,N);

    err = cudaMemcpy((void*)B, d_B, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
    printf("Computing Serially.\n");

}

void print_inputs() {
    int row, col;

    if (N < 10) {
        printf("\nA =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }

        printf("\nB =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%5.2f%s", B[row][col], (col < N-1) ? ", " : ";\n\t");
            }


        }
    }
}

int main(int argc, char **argv) {
    /* Timing variables */
    struct timeval start, stop;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    unsigned long long runtime;

    /* Initialize A and B */
    initialize_inputs();


    /* Start Clock */
    printf("\n---------------------------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");
    gettimeofday(&start, &tzdummy);


    /* Matrix Normalization */
    matrixNorm();
    print_inputs();


    /* Stop Clock */
    gettimeofday(&stop, &tzdummy);
    runtime = (unsigned long long)(stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);


    /* Display timing results */
    printf("Runtime = %g ms.\n", (float)runtime/(float)1000);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");

    exit(0);
}
