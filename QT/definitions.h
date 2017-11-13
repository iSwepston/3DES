#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <cuda_runtime.h>
#include <des_constants.h>

// MACROS
#define GET_RDTSC(lo,hi,time) {\
    __asm__ volatile("rdtsc" : "=a" (lo), "=d" (hi));\
    time = ((uint64_t)hi << 32) | lo;\
    }

// for simplicity
typedef unsigned char byte;

__global__ void cuPC1(byte *key, byte *result);
__device__ byte doShifting(int * shiftArray, byte * input, int id);

__device__ int get_idx()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

#endif // DEFINITIONS_H
