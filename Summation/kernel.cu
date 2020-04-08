#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../Common/common_tools.h"

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using Clock = std::chrono::high_resolution_clock;
using SecondTick = std::chrono::duration<long double>;

__global__ void sum_array_gpu(int *a, int *b, int *c, int size) {

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    c[gid] = a[gid] + b[gid];
}

void sum_array_cpu(int *a, int *b, int *c, int size) {

  for (int i = 0; i < size; i++)
    c[i] = a[i] + b[i];
}

void compare_arrays(int *a, int *b, int size) {

  for (int i = 0; i < size; i++) {
    if (a[i] != b[i]) {
      printf("Arrays are different: a[%d] : %d != b[%d] : %d \n", i, a[i], i, b[i]);
      return;
    }
  }
  printf("Arrays are same.\n");
}

int main() {
  int SIZE = (int)1e8;
  int BLOCK_SIZE = 2^10;
  int NUM_BYTES = SIZE * sizeof(int);

  // Host pointers
  int *h_a = (int *)malloc(NUM_BYTES);
  int *h_b = (int *)malloc(NUM_BYTES);
  int *h_c1 = (int *)malloc(NUM_BYTES);
  int *h_c2 = (int *)malloc(NUM_BYTES);

  // Fill data with random numbers
  time_t t;
  srand((unsigned)time(&t));
  for (size_t i = 0; i < SIZE; i++) {
    h_a[i] = (int)(rand() & 0xFF);
    h_b[i] = (int)(rand() & 0xFF);
  }

  // First try cpu implementation
  auto cpu_start = Clock::now();
  sum_array_cpu(h_a, h_b, h_c1, SIZE);
  auto cpu_end = Clock::now();

  // Now try gpu implementation - initialize device pointers
  int *d_a, *d_b, *d_c;
  gpuErrchk(cudaMalloc((int **)&d_a, NUM_BYTES));
  gpuErrchk(cudaMalloc((int **)&d_b, NUM_BYTES));
  gpuErrchk(cudaMalloc((int **)&d_c, NUM_BYTES));

  // Memory transfer from host to device
  auto htod_start = Clock::now();
  gpuErrchk(cudaMemcpy(d_a, h_a, NUM_BYTES, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_b, h_b, NUM_BYTES, cudaMemcpyHostToDevice));
  auto htod_end = Clock::now();

  // Create and launch kernel
  dim3 block(BLOCK_SIZE);
  dim3 grid(SIZE / block.x);

  auto gpu_start = Clock::now();
  sum_array_gpu<<<grid, block>>>(d_a, d_b, d_c, SIZE);
  gpuErrchk(cudaDeviceSynchronize());
  auto gpu_end = Clock::now();

  // Memory transfer from device to host
  auto dtoh_start = Clock::now();
  gpuErrchk(cudaMemcpy(h_c2, d_c, NUM_BYTES, cudaMemcpyDeviceToHost));
  auto dtoh_end = Clock::now();

  // compare arrays to check if they are equivalent
  compare_arrays(h_c1, h_c2, SIZE);

  // cleanup
  gpuErrchk(cudaFree(d_c));
  gpuErrchk(cudaFree(d_b));
  gpuErrchk(cudaFree(d_a));

  free(h_c2);
  free(h_c1);
  free(h_b);
  free(h_a);

  // Log time for cpu and gpu implementations
  long double elapsedTimeCpu = SecondTick(cpu_end - cpu_start).count();
  printf("CPU implementation time      = %.2fs\n", elapsedTimeCpu);

  long double elapsedTimeGpu = SecondTick(gpu_end - gpu_start).count();
  printf("GPU implementation time      = %.2fs\n", elapsedTimeGpu);

  long double elapsedTimeHtod = SecondTick(htod_end - htod_start).count();
  printf("Host to device transfer time = %.2fs\n", elapsedTimeHtod);

  long double elapsedTimeDtoh = SecondTick(dtoh_end - dtoh_start).count();
  printf("Device to host transfer time = %.2fs\n", elapsedTimeDtoh);

  long double elapsedTimeGpuTotal = elapsedTimeGpu + elapsedTimeHtod + elapsedTimeDtoh;
  printf("Total GPU implementaion time = %.2fs\n", elapsedTimeGpuTotal);
}