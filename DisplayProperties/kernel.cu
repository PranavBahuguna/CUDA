#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void query_device() {

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0)
    printf("No CUDA support devices found!\n");

  for (int devNo = 0; devNo < deviceCount; devNo++) {
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);

    printf("Device %d: %s\n", devNo + 1, iProp.name);
    printf("  Number of multiprocessors:                     %d\n", iProp.multiProcessorCount);
    printf("  Clock rate:                                    %d\n", iProp.clockRate);
    printf("  Compute capability:                            %d.%d\n", iProp.major, iProp.minor);
    printf("  Total amount of global memory:                 %4.2f KB\n",
           (double)(iProp.totalGlobalMem / (1 << 10)));
    printf("  Total amount of constant memory:               %4.2f KB\n",
           (double)(iProp.totalConstMem / (1 << 10)));
    printf("  Total amount of shared memory per block:       %4.2f KB\n",
           (double)(iProp.sharedMemPerBlock / (1 << 10)));
    printf("  Total amount of registers available per block: %4.2f KB\n",
           (double)(iProp.regsPerBlock / (1 << 10)));
    printf("  Total amount of global memory:                 %4.2f KB\n",
           (double)(iProp.totalGlobalMem / (1 << 10)));
    printf("  Warp size:                                     %d\n", iProp.warpSize);
    printf("  Maximum number of threads per block:           %d\n", iProp.maxThreadsPerBlock);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           iProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of warps per multiprocessor:    %d\n",
           iProp.warpSize / iProp.maxThreadsPerMultiProcessor);
    printf("  Maximum grid size:                             (%d, %d, %d)\n", iProp.maxGridSize[0],
           iProp.maxGridSize[1], iProp.maxGridSize[2]);
    printf("  Maximum block size:                            (%d, %d, %d)\n\n",
           iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);
  }
}

int main() {
  query_device();
  return 0;
}