#include <iostream>
#include <math.h>
#include <chrono>

// helper functions
void random_ints(int* array, int array_length) {
    for (int i = 0; i < array_length; ++i) {
        array[i] = rand() % 100; // random integer between 0 and 99
    }
}

// a simple program for adding two arrays without the use of CUDA
void add(int *a, int *b, int *c, long N) {
    // each parallel invocation of add is referred to as a block - set of all blocks is a grid
    // each invocation can refer to its block index using blockIdx.x
    // c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];

    // we can alternative split a block into parallel threads and have our program run parallel threads instead of parallel blocks
    for (long i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

#define N 1073741824 // 1B+ elements
int main(void) {
    int *a, *b, *c;

    long size = N * sizeof(int); // need to allocate space for device copies of a, b, c

    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);

    c = (int *)malloc(size);

    auto start = std::chrono::high_resolution_clock::now();

    // triple brackets mark a call to device code or "kernel launch"
    // inner parameters are execution configuration - first parameter indicates executing add a number of times
    add(a, b, c, N);

    // cleanup
    free(a); free(b); free(c);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Added two " << N << "-integer vectors in " << duration.count() << " microseconds" << std::endl;

    return 0;
}