/* Matrix normalization.
 * Compile with "gcc matrixNorm.c"
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
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

/* Initialize A and B */
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

/* colSum */
__global__ void colSum(float* input_mat, float* sum_vec, int n, int col)
{
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < n) 
    {
        sdata[tid] = input_mat[row * n + col];
        __syncthreads();

    }

    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0 && tid + s < n)
        {
            sdata[tid] += sdata[tid + s];
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        sum_vec[blockIdx.x * n + col] = sdata[0];
    }
}

/* vecDiv */
__global__ void vecDiv(float* input_vec, int n, float x)
{
    //i 每一个线程的id
    //n 向量长度
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        input_vec[i] /= x;
    }
}

/* vecSqrt */
__global__ void vecSqrt(float* input_vec, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        input_vec[i] = sqrt(input_vec[i]);
    }
}

/* getSquareError */
__global__ void getSquareError(float* input_mat, float* output_mat, float* mean_vec, int n)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col < n && row < n)
    {
        float tmp = input_mat[row * n + col] - mean_vec[col];
        output_mat[row * n + col] = tmp * tmp;
    }
}

/* getZScore */
__global__ void getZScore(float* input_mat, float* output_mat, float* mean_vec, float* sigma_vec, int n)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (col < n && row < n)
    {
        if (sigma_vec[col] == 0)
        {
            output_mat[row * n + col] = 0.0;
        }
        else
        {
            output_mat[row * n + col] = (input_mat[row * n + col] - mean_vec[col]) / sigma_vec[col];
        }
    }
}

/* getColMean */
void getColMean (float* d_A, float* mean_vec, int N)
{
    for (int col = 0; col < N; ++col)
    {
        //kernel 1
        int blockNum = ceil((float)N / 256.0);
        if (blockNum > 1)
        {
            float* temp;
            cudaError_t err = cudaMalloc((void **) &temp, sizeof(float) * blockNum * N);
            CHECK_ERR(err);

            colSum<<<blockNum, 256, sizeof(float) * 256>>>(d_A, temp, N, col);
            int blockNum1 = ceil((float)blockNum / 256.0);
            assert(blockNum1 == 1);
            colSum<<<blockNum1, 256, sizeof(float) * 256>>>(temp, mean_vec, blockNum, col);
        } else {
            colSum<<<blockNum, 256, sizeof(float) * 256>>>(d_A, mean_vec, N, col);
        }
        vecDiv<<<blockNum, 256>>>(mean_vec, N, N);
    }
}

/* Kernel function */

void matrixNorm() {

    cudaError_t err;
    //start from here:
    float *d_A;
    float *d_B; 
    float *mean_vec; 
    float *sigma_vec; 
    float *tmp;

    err = cudaMalloc((void **) &d_A, sizeof(float)*N*N);
       CHECK_ERR(err);
    err = cudaMalloc((void **) &d_B, sizeof(float)*N*N);
       CHECK_ERR(err);
    err = cudaMalloc((void **) &tmp, sizeof(float)*N*N);
       CHECK_ERR(err);
    err = cudaMalloc((void **) &mean_vec, sizeof(float) * N);
       CHECK_ERR(err);
    err = cudaMalloc((void **) &sigma_vec, sizeof(float) * N);
       CHECK_ERR(err);
    err = cudaMemcpy(d_A, (void*)A , sizeof(float)*N*N, cudaMemcpyHostToDevice);
        CHECK_ERR(err);
    //kernal function here

    //share memory allocating
    getColMean(d_A, mean_vec, N);
    {
        int blockDim = ceil((float)N / 16.0);
        getSquareError<<<{blockDim, blockDim}, {16, 16}>>>(d_A, tmp, mean_vec, N);
    }
    getColMean(tmp, sigma_vec, N);
    {
        int blockDim = ceil((float)N / 256.0);
        vecSqrt<<<blockDim, 256>>>(sigma_vec, N);
    }
    {
        int blockDim = ceil((float)N / 16.0);
        getZScore<<<{blockDim, blockDim}, {16, 16}>>>(d_A, d_B, mean_vec, sigma_vec, N);
    }
    err = cudaMemcpy((void*)B, d_B, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
    printf("Computing Serially.\n");

    //free all pointers
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(*mean_vec);
    cudaFree(*sigma_vec);
    cudaFree(*tmp);
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
