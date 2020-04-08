
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_cuda() {
  printf("threadIdx = (%d, %d, %d), blockIdx = (%d, %d, %d)\n", threadIdx.x, threadIdx.y,
         threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}

int main() {

  int nx = 16;
  int ny = 4;

  dim3 block(8, 2);
  dim3 grid(nx / block.x, ny / block.y);

  hello_cuda<<<grid, block>>>();
  cudaDeviceSynchronize();

  cudaDeviceReset();
  return 0;
}