#include <ctime>
#include <iostream>
#include "cuda_runtime.h"
#include "error.cuh"
#include "device.cuh"
#include "timer.cuh"

using namespace std;
#define N 1024

void matrix_multiply_normal(float * A, float * B, float * C) {
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			C[i * N + j] = 0;
			for (int k = 0; k < N; k++)
				C[i * N + j] += A[i * N + k] * B[k * N + j];
		}
}

__global__ void matrix_multiply_normal_cuda(float * A, float * B, float * C) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N * N / 4) {
		int j = i % N;
		i = i / N;
		C[j * N + i * 4] = C[j * N + i * 4 + 1] = C[j * N + i * 4 + 2] = C[j * N + i * 4 + 3] = 0;
		for (int k = 0; k < N; k++) {
			C[j * N + i * 4] += A[j * N + k] * B[k * N + i * 4];
			C[j * N + i * 4 + 1] += A[j * N + k] * B[k * N + i * 4 + 1];
			C[j * N + i * 4 + 2] += A[j * N + k] * B[k * N + i * 4 + 2];
			C[j * N + i * 4 + 3] += A[j * N + k] * B[k * N + i * 4 + 3];
		}
	}
	/*
	if (i < N * N) {
		C[i] = 0;
		for (int k = 0; k < N; k++)
			C[i] += A[(i / N) * N + k] * B[k * N + i % N];
	}*/
}

int main() {
	size_t size = sizeof(float) * N * N;
	float * matrixA = (float *)malloc(size);
	float * matrixB = (float *)malloc(size);
	float * matrixC = (float *)malloc(size);
	float * matrixD = (float *)malloc(size);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			matrixA[i * N + j] = j % (N / 32);
			matrixB[i * N + j] = i % (N / 32);
		}
	}

	{
		auto timer = CudaTimer();
		float *gpu_matrixA, *gpu_matrixB, *gpu_matrixC;
		CUDA_CHECK(cudaMalloc(&gpu_matrixA, size));
		CUDA_CHECK(cudaMalloc(&gpu_matrixB, size));
		CUDA_CHECK(cudaMalloc(&gpu_matrixC, size));
		CUDA_CHECK(cudaMemcpy(gpu_matrixA, matrixA, size, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(gpu_matrixB, matrixB, size, cudaMemcpyHostToDevice));

		//dim3 gridBlocks(32, 32);
		//dim3 blockThreads(32, 32);
		matrix_multiply_normal_cuda <<<N / 4, N>>> (gpu_matrixA, gpu_matrixB, gpu_matrixC);
		//matrix_multiply_normal_cuda <<<N, N>>> (gpu_matrixA, gpu_matrixB, gpu_matrixC);
		CUDA_CHECK_ERROR("kernel panic!!!");
		CUDA_CHECK(cudaMemcpy(matrixC, gpu_matrixC, size, cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaFree(gpu_matrixA));
		CUDA_CHECK(cudaFree(gpu_matrixB));
		CUDA_CHECK(cudaFree(gpu_matrixC));
	}

	/*
	{
		auto timer = CpuTimer();
		matrix_multiply_normal(matrixA, matrixB, matrixD);
	}
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			if (matrixC[i * N + j] != matrixD[i * N + j])
			{
				cout << matrixC[i * N + j] << " " << matrixD[i * N + j];
				cout << "wrong ! " << i << endl;
				break;
			}
	*/
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	return 0;
}