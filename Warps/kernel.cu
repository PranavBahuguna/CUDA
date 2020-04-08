#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void print_warps_details() {

  int gbid = blockIdx.y * gridDim.x + blockIdx.x;
  int gid = gbid * blockDim.x + threadIdx.x;
  int wid = threadIdx.x / 32;

  printf("tid : %d, bid : [%d, %d], gid : %d,  wid : %d, gbid : %d\n", threadIdx.x, blockIdx.x,
         blockIdx.y, gid, wid, gbid);
}

int main(int argc, char *argv[]) { 
  
  dim3 block_size(42);  // 10 larger than actual warp size
  dim3 grid_size(2, 2);

  print_warps_details<<<grid_size, block_size>>>();
  cudaDeviceSynchronize();

  cudaDeviceReset();
  return EXIT_SUCCESS;
}