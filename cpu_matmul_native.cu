#include <iostream>
#include <assert.h>
#include "matrix.cuh"

void cpu_matmul(const Matrix A, const Matrix B, Matrix C) {
    assert(A.height == B.width);
    assert(A.width == C.width && B.height == C.height);
    for (int i = 0; i < A.width; i++)
        for (int j = 0; j < B.height; j++) {
            float value = 0;
            for (int k = 0; k < A.height; k++) 
                value += getElement(A, i, k) * getElement(B, k, j);
            setElement(C, i, j, value);
        }
}