#pragma once

typedef struct {
    int width;
    int height;
    int stride;
    float * elements;
} Matrix;

__forceinline__ 
float getElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

__forceinline__
void setElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}

__forceinline__
Matrix getSubMatrix(Matrix A, int row, int col, int block_size) {
    Matrix Asub;
    Asub.width = block_size;
    Asub.height = block_size;
    Asub.stride = A.stride;
    int address = A.stride * block_size * row + block_size * col;
    Asub.elements = &A.elements[address];
    return Asub;
}

__device__ __forceinline__ 
float getElementGPU(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

__device__ __forceinline__
void setElementGPU(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}

__device__ __forceinline__
Matrix getSubMatrixGPU(Matrix A, int row, int col, int block_size) {
    Matrix Asub;
    Asub.width = block_size;
    Asub.height = block_size;
    Asub.stride = A.stride;
    int address = A.stride * block_size * row + block_size * col;
    Asub.elements = &A.elements[address];
    return Asub;
}

bool checkMatrixSame(const Matrix A, const Matrix B) {
    if (A.width != B.width or A.height != B.height) return false;
    if (A.stride != B.stride) return false;
    for (int i = 0; i < A.width; i++)
        for (int j = 0; j < A.height; j++)
            if (getElement(A, i, j) != getElement(B, i, j))    
                return false;
    return true;
}