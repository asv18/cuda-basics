#include <stdio.h>

__global__ void hello() {
  printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

int main() {
  hello<<<2, 2>>>();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
  }

  return 0;
}

