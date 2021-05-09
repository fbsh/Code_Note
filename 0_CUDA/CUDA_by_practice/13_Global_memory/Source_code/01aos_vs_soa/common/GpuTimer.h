#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

//#include <cuda_runtime.h>

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() { cudaEventRecord(start, 0); }

    void Stop() { cudaEventRecord(stop, 0); }

    void Start(cudaStream_t stream) { cudaEventRecord(start, stream); }

    void Stop(cudaStream_t stream) { cudaEventRecord(stop, stream); }

    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        int counter = 0;
        while (cudaEventQuery(stop) == cudaErrorNotReady) {
            counter++;
            printf("Waiting for the Device to finish: %d\n", counter);
        }
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

#endif /* GPU_TIMER_H__ */
