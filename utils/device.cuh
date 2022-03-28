#pragma once
#include <stdio.h>
#include "error.cuh" 

void deviceProperties() {
    cudaDeviceProp prop;
    int count, driverVersion = 0, runtimeVersion = 0;
    cudaGetDeviceCount(&count);
    CUDA_CHECK_ERROR("device count");

    for (int i = 0; i < count; i++) {
        cudaGetDeviceProperties(&prop, i);
        CUDA_CHECK_ERROR("device prop");
        std::printf(" -- General Information for device %d --\n", i);
        std::printf("Name:  %s\n", prop.name);
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        std::printf("CUDA Driver Version  %d.%d\n",
                    driverVersion / 1000, (driverVersion % 100) / 10);
        std::printf("Runtime Version %d.%d\n",
                  runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        std::printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        std::printf("\n");
        std::printf("Clock rate:  %.0f MHz (%.0f GHz)\n",
                  prop.clockRate * 1e-3f, prop.clockRate * 1e-6f);  
        std::printf("\n");
        std::printf("Concurrent kernels:  %s \n",
                  prop.concurrentKernels ? "Enabled" : "Disabled");
#if CUDART_VERSION >= 5000
        std::printf("Concurrent copy and kernel execution %s with %d copy engine(s)\n",
                    (prop.deviceOverlap ? "Enabled" : "Disabled"),
                     prop.asyncEngineCount);
#endif
        std::printf("Kernel execution timeout :  %s \n",
                     prop.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");
        std::printf("Integrated GPU sharing Host Memory: %s\n",
                     prop.integrated ? "Enabled" : "Disabled");
        std::printf("Support host page-locked memory mapping: %s\n",
                     prop.canMapHostMemory ? "Enabled" : "NoDisabled");
        std::printf("\n   --- Memory Information for device %d ---\n", i);
#if CUDART_VERSION >= 5000
        std::printf("Memory Clock rate: %f Ghz\n", prop.memoryClockRate * 10e-7);
        std::printf("Memory Bus Width:  %d-bit\n", prop.memoryBusWidth);
#endif
        std::printf("Total global mem:  %lf Mbytes (%ld bytes) \n",
                     prop.totalGlobalMem / 1048576.0, prop.totalGlobalMem);
        std::printf("Total constant Mem:  %ld bytes\n", prop.totalConstMem);
        std::printf("Max mem pitch:  %ld bytes\n", prop.memPitch);
        std::printf("\n   --- MP Information for device %d ---\n", i);
        std::printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
        std::printf("Shared mem per block:  %ld bytes \n", prop.sharedMemPerBlock);
        std::printf("Registers per block:  %d\n", prop.regsPerBlock);
        std::printf("Threads in warp:  %d\n", prop.warpSize);
#if CUDART_VERSION >= 5000
        std::printf("Max threads per multiprocessor: %d\n",
                     prop.maxThreadsPerMultiProcessor);
#endif
        std::printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
        std::printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0],
                     prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        std::printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0],
                     prop.maxGridSize[1], prop.maxGridSize[2]);
        std::printf("\n");
    }
}