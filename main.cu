#include <ctime>
#include <iostream>
#include "cuda_runtime.h"
#include "error.cuh"
#include "device.cuh"
#include "timer.cuh"
#include "matrix.cuh"
#include "gpu_matmul_block.cu"
#include "cpu_matmul_native.cu"

using namespace std;
#define N 1024

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

    Matrix A, B, C, D;
    A.width = A.height = B.width = B.height = C.width = C.height = N;
    A.stride = B.stride = C.stride = N;
    D.width = D.height = D.stride = N;

	float *gpu_matrixA, *gpu_matrixB, *gpu_matrixC;
	CUDA_CHECK(cudaMalloc(&gpu_matrixA, size));
	CUDA_CHECK(cudaMalloc(&gpu_matrixB, size));
	CUDA_CHECK(cudaMalloc(&gpu_matrixC, size));
	CUDA_CHECK(cudaMemcpy(gpu_matrixA, matrixA, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(gpu_matrixB, matrixB, size, cudaMemcpyHostToDevice));

    A.elements = gpu_matrixA;
    B.elements = gpu_matrixB;
    C.elements = gpu_matrixC;

	{
		auto timer = CudaTimer();
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
		for (int i = 0; i < 10; ++i) {
            gpu_matmul_block<<<dimGrid, dimBlock>>>(A, B, C);
		}
	}
	
    CUDA_CHECK_ERROR("kernel panic!!!");
	CUDA_CHECK(cudaMemcpy(matrixC, gpu_matrixC, size, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(gpu_matrixA));
	CUDA_CHECK(cudaFree(gpu_matrixB));
	CUDA_CHECK(cudaFree(gpu_matrixC));

    A.elements = matrixA;
    B.elements = matrixB;
    C.elements = matrixD;
    D.elements = matrixC;
    {
        auto timer = CpuTimer();
        cpu_matmul(A, B, C);
    }
    std::cout << checkMatrixSame(C, D);
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	return 0;
}
