// System includes
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <algorithm>

//#include <math_constants.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <math_constants.h>

// libs
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

// User includes
#include <des_constants.h>

// MACROS
#define GET_RDTSC(lo,hi,time) {\
    __asm__ volatile("rdtsc" : "=a" (lo), "=d" (hi));\
    time = ((uint64_t)hi << 32) | lo;\
    }

typedef unsigned char byte;

// Goes from 8 bytes downto 7
__global__ void cuPC1(byte *key, byte *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    byte output = 0x00;

    for(int i = id * 8; i < (id+1)*8; i++) {
        // determine bit to relocate
        int bit = PC_1[i] - 1;

        byte temp = key[bit/8]; // get approprate byte
        if(id == 0) printf("Key section: %x\n", temp);

        temp = temp >> (bit+1)%8; // shift to position 0
        temp &= 0x01; // mask in only this bit

        if(id == 0) printf("%d, bit %d, res: %x\n", id, bit+1, temp);

        temp = temp << i%8; // shift to right position

        output |= temp;
    }

    result[id] = output;
}


int main(int argc, char **argv)
{

    std::cout << "Testing" << std::endl;

    byte * inputKey;
    byte * output;
    cudaMalloc((void**)&inputKey, sizeof(byte)*8);
    cudaMalloc((void**)&output, sizeof(byte)*7);

    byte key[8] = {0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10};

    cudaMemcpy(inputKey, &key, sizeof(uint64_t), cudaMemcpyHostToDevice);

    cuPC1<<<1,7>>>(inputKey, output);

    cudaDeviceSynchronize();

    byte result[7];
    cudaMemcpy(result, output, 7, cudaMemcpyHostToDevice);

    printf("Output: ");
    for(int i = 0; i < 7; i++)
        printf("%X",result[i]);

    printf("\n");

    return 0;
}
