#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void memTransfer(int* input) {
  int gID = blockIdx.x * blockDim.x + threadIdx.x;
  printf("tid = %d, gid = %d, value = %d\n", threadIdx.x, gID, input[gID]);
}

int main() {

  int size = 128;
  int byteSize = size * sizeof(int);

  int *sInput;
  sInput = (int *)malloc(byteSize);

  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; ++i) {
    sInput[i] = (int)(rand() & 0xff);
  }

  int *dInput;
  cudaMalloc((void **)&dInput, byteSize);

  cudaMemcpy(dInput, sInput, byteSize, cudaMemcpyHostToDevice);

  dim3 block(64);
  dim3 grid(2);

  memTransfer<<<grid, block>>>(dInput);
  cudaDeviceSynchronize();

  cudaFree(dInput);
  free(sInput);

  cudaDeviceReset();
  return 0;
}