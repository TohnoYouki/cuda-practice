#include "matrix.cuh"

#define BLOCK_SIZE 32

__global__ void gpu_matmul_block(Matrix A, Matrix B, Matrix C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    float value = 0.0;
    Matrix CSub = getSubMatrixGPU(C, blockRow, blockCol, BLOCK_SIZE);

    for (int i = 0; i < (A.height / BLOCK_SIZE); i++) {
        Matrix ASub = getSubMatrixGPU(A, blockRow, i, BLOCK_SIZE);
        Matrix BSub = getSubMatrixGPU(B, i, blockCol, BLOCK_SIZE);
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[row][col] = getElementGPU(ASub, row, col);
        Bs[row][col] = getElementGPU(BSub, row, col);
        __syncthreads();
        for (int j = 0; j < BLOCK_SIZE; j++)
            value += As[row][j] * Bs[j][col];
        __syncthreads();
    }
    setElementGPU(CSub, row, col, value);
}