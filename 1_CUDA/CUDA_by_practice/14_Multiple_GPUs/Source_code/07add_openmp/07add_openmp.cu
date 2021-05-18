#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include "common/CpuTimer.h"
#include "common/Vector.h"

#define THREADS 2
#define N (32)

const int ARRAY_BYTES = N * sizeof(float);
const int ARRAY_BYTES_P = N / THREADS * sizeof(float);

struct DataStruct {
    int threadID;
    int size;
    float* a;
    float* b;
    float* out;
};

void test(Vector<float> h_sout, Vector<float> h_mout) {
    for (int i = 0; i < N; i++) {
        assert(h_sout.elements[i] == h_mout.elements[i]);
    }
}

void threadRoutine(void* param) {
    DataStruct* data = (DataStruct*)param;

    printf("ThreadID = %d \n", data->threadID);

    for (int i = 0; i < data->size; i++) {
        data->out[i] = data->a[i] + data->b[i];
    }
}

void addMutiple(Vector<float> h_a, Vector<float> h_b, Vector<float> h_mout) {
    // prepare for multithread
    DataStruct data[THREADS];
    CpuTimer timer;
    timer.Start();

    for (int i = 0; i < THREADS; i++) {
        data[i].threadID = i;
        data[i].size = N / THREADS;
        data[i].a = h_a.elements + N / THREADS * i;
        data[i].b = h_b.elements + N / THREADS * i;
        data[i].out = (float*)malloc(ARRAY_BYTES_P);
    }

    for (int i = 0; i < THREADS; i++) {
        for (int j = 0; j < N / THREADS; j++) {
            data[i].out[j] = 0;
        }
    }

    omp_set_num_threads(THREADS);
#pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        threadRoutine(&data[cpu_thread_id]);
    }

    for (int i = 0; i < THREADS; i++) {
        for (int j = 0; j < data[i].size; j++) {
            h_mout.elements[N / THREADS * i + j] = data[i].out[j];
        }
    }

    timer.Stop();
    printf("CPU Time :  %f ms\n", timer.Elapsed());
}

void addSingle(Vector<float> h_a, Vector<float> h_b, Vector<float> h_sout) {
    for (int i = 0; i < N; i++) {
        h_sout.setElement(i, h_a.getElement(i) + h_b.getElement(i));
    }
}

void run() {
    Vector<float> h_a, h_b, h_sout, h_mout;
    h_a.length = N;
    h_b.length = N;
    h_sout.length = N;
    h_mout.length = N;

    h_a.elements = (float*)malloc(ARRAY_BYTES);
    h_b.elements = (float*)malloc(ARRAY_BYTES);
    h_sout.elements = (float*)malloc(ARRAY_BYTES);
    h_mout.elements = (float*)malloc(ARRAY_BYTES);

    for (int i = 0; i < N; i++) {
        h_a.elements[i] = i;
        h_b.elements[i] = i;
        h_sout.elements[i] = 0;
        h_mout.elements[i] = 0;
    }

    addSingle(h_a, h_b, h_sout);
    addMutiple(h_a, h_b, h_mout);

    test(h_sout, h_mout);
    printf("-: successful execution :-\n");
}

int main(void) {
    run();
    return 0;
}
