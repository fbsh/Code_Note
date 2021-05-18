__global__ void 
Reduction1_kernel( int *out, const int *in, size_t N ) 
{ 
    extern __shared__ int sPartials[];
    int sum = 0;
    const int tid = threadIdx.x; 
    for ( size_t i = blockIdx.x*blockDim.x + tid; i < N; i += blockDim.x*gridDim.x ) {
        sum += in[i];
    }
    sPartials[tid] = sum;
    __syncthreads();
    
    for ( int activeThreads = blockDim.x>>1; activeThreads; activeThreads >>= 1 ) { 
        if ( tid < activeThreads ) {
            sPartials[tid] += sPartials[tid+activeThreads];
            } 
        __syncthreads();
    }
    if ( tid == 0 ) {
        out[blockIdx.x] = sPartials[0];
    }
}

void 
Reduction1(int *answer, int *partial, const int *in, size_t N, int numBlocks, int numThreads) {
    unsigned int sharedSize = numThreads*sizeof(int);
    Reduction1_kernel<<< numBlocks, numThreads, sharedSize>>>(partial, in, N);
    Reduction1_kernel<<<1, numThreads, sharedSize>>>(answer, partial, numBlocks);
}