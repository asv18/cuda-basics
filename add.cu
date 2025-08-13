#include <iostream>
#include <math.h>
#include <chrono>

// helper functions
void random_ints(int* array, int array_length) {
    for (int i = 0; i < array_length; ++i) {
        array[i] = rand() % 100; // random integer between 0 and 99
    }
}

// a simple program for adding two arrays

// __global__ indicates a function that runs on the GPU and is called from the host code
// nvcc separates source code into host and device components
__global__ void add(int *a, int *b, int *c) {
    // each parallel invocation of add is referred to as a block - set of all blocks is a grid
    // each invocation can refer to its block index using blockIdx.x
    // c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];

    // we can alternatively split a block into parallel threads and have our program run parallel threads instead of parallel blocks
    // c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];

    // one way to make use of both threads and blocks is by using both the blockIdx and threadIdx together
    // for example, if for every block index we have 8 thread indices, for M threads/blocks, the corresponding index would be:
    // int index = threadIdx.x + blockIdx.x * M;
    // we do not have to pass M, however - the builtin variable blockDim.x gives us the value we need (threads/block)
    long index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] + b[index];
}

#define N 1073741824 // define num elements
#define THREADS_PER_BLOCK 16384 // define the number of threads per block
int main(void) {
    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c

    long size = N * sizeof(int); // need to allocate space for device copies of a, b, c
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);

    c = (int *)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    // triple brackets mark a call to device code or "kernel launch"
    // inner parameters are execution configuration - first parameter indicates using multiple blocks, second parameter indicates using multiple threads
    // we can make our programs even more massively parallel by using both blocks and threads in tandem 
    add<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    // threads are useful as, unlike blocks, they have mechanisms to communicate and synchronize

    // copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Added two " << N << "-integer vectors in " << duration.count() << " microseconds" << std::endl;

    return 0;
}