#pragma once
#include <ctime>
#include <iostream>
#include "error.cuh"

class CpuTimer {
private:
	clock_t start, stop;
public:
	CpuTimer() {
		start = clock();
	}

	~CpuTimer() {
		stop = clock();
		double time = 0.0;
		time = double(stop - start) * 1000 / CLOCKS_PER_SEC;
		std::cout << "Time: " << time << " ms" << std::endl;
	}
};

class CudaTimer {
private:
	cudaEvent_t start, stop;
public:
	CudaTimer() { 
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
	}

	~CudaTimer() {
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		while (cudaEventQuery(stop) == cudaErrorNotReady);
		float time;
		cudaEventElapsedTime(&time, start, stop);
		std::cout << "Time: " << time << " ms" << std::endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
};