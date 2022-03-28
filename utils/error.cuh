#pragma once
#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"

#define CUDA_CHECK(status)                                       \
	do {                                                         \
		cudaError_t status_ = status;                            \
		if (cudaSuccess != status_) {                            \
			auto msg = cudaGetErrorString(status_);              \
			std::printf("failed with status %d: %s at %s:%d\n",  \
                         status_, msg, __FILE__, __LINE__);      \
			throw std::logic_error("cuda API failed!");          \
		}                                                        \
	} while (0)


#define CUDA_CHECK_ERROR(msg)                                    \
	do {                                                         \
		cudaError_t status = cudaGetLastError();                 \
		if (cudaSuccess != status) {                             \
			auto error_msg = cudaGetErrorString(status);         \
			std::printf("%s: %s in %s at line %d\n",             \
                        msg, error_msg, __FILE__, __LINE__);     \
			throw std::logic_error("cuda Logic Error!");         \
		}                                                        \
	} while (0)