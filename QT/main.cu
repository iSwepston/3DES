// System includes
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <unistd.h>
#include <stdint.h>
#include <algorithm>

// CUDA runtime
#include <cuda_runtime.h>
#include <math_constants.h>

// User includes
#include <definitions.h>

/**
 * @brief doShifting: a device function which is designed to do the "work" of masking a byte
 * @param shiftArray: The shifting directions
 * @param input: input byte array array to shift around
 * @return
 */
__device__ byte doShifting(int * shiftArray, byte * input, int id)
{
    byte output = 0x00;

    // Loop over all bits in current byte
    for(int i = id * 8; i < (id+1)*8; i++) {

        // determine bit to relocate
        int bit = shiftArray[i] - 1;

        // mask and shift
        byte temp = input[bit/8]; // get approprate byte
        temp = temp >> (8-(bit+1)%8)%8; // shift to position 0
        temp &= 0x01; // mask in only this bit

        temp = temp << 7 - i%8; // shift to right position

        // set bits
        output |= temp;
    }

    return output;
}

// Goes from 8 bytes downto 7
__global__ void cuPC1(byte *key, byte *result)
{
    int id = get_idx();

    result[id] = doShifting(PC_1, key, id);
}

// one thread per key
__global__ void cuLeftCircShifts(byte * input, byte * result)
{
    int id = get_idx();
    int shamt = id + 1;

    uint upper = 0;
    uint lower = 0;

    for(int i = 0; i <= 3; i++) {
        uint temp = 0;
        temp += input[i];

        temp = temp << (3-i)*8;
        upper |= temp;
    }

    upper = upper >> 4;

    for(int i = 3; i <= 6; i++) {
        uint temp = 0;
        temp |= input[i];
        temp = temp << (6 - i)*8;
        lower |= temp;
    }

    // mask out
    lower &= 0x0FFFFFFF;

    int numbits = 28; //7 nibbles
    uint mask = upper >> numbits - shamt;
    upper = upper << shamt;
    upper |= mask;
    upper &= 0x0FFFFFFF;

    mask = lower >> numbits - shamt;
    lower = lower << shamt;
    lower |= mask;
    lower &= 0x0FFFFFFF;

    uint64_t out = upper;
    out <<= numbits;
    out |= lower;

    int start = id*7;
    for(int i = start; i < start+7; i++) {
        result[i] = (out >> (6 - i%7)*8) & 0x0FF;
    }

    //    printf("id %d: %lx\n", id, out);
}

__global__ void cuPC2(byte * input, byte * round_keys)
{
    int id = get_idx();
    int round = (id/6);

    //    byte * x = &input[round*7];
    //    for(int i = 0; i < 7; i++) {
    //        if(id==7) printf("%02X\n", x[i]);
    //    }

    byte temp = doShifting(PC_2, &input[round*7], id%6);
    //    if(id == 7) printf("Res: %0x\n", temp);
    round_keys[id] = temp;
}

// 64 -> 64
__global__ void cuIP(byte * input, byte * output)
{
    int id = get_idx();

    output[id] = doShifting(IP, input, id);

    printf("IP: id = %d: %02x\n", id, output[id]);
}

// 32(64) -> 48
__global__ void cuEPerm(byte * input, byte * output)
{
    int id = get_idx();

    output[id] = doShifting(Ex, input, id%3);

    printf("Ex: id = %d: %02x\n", id, output[id]);
}

// 48 -> 48
__global__ void cuXOR(byte * input, byte * key, byte * output)
{
    int id = get_idx();

    output[id] = input[id] ^ key[id%6];

    printf("XOR: id = %d: %02x\n", id, output[id]);
}

// 48 -> 32
__global__ void cuSBoxes(byte * input, byte * output)
{
    int id = get_idx();
    output[id] = 0;

    uint64_t value = 0;
    for(int i = 0; i < 6; i++) {
        value |= input[i];
        value = value << 8;
    }

    value = value >> 8;

    int modId = id % 8;
    byte test = (value >> (7 - modId)*6) & 0x3F;
    byte row = ((test >> 4) & 0x02) | (test & 0x01); // outer bits
    byte col = (test >> 1) & 0x0F; // inner bits

//    if (id == 0) printf("%lx\n", value);

    int offset = row * 16 + col;
    byte result = boxes[id%8][offset];

    output[id] |= result << 4*(1 - (id % 2));
    __syncthreads();

    if(id % 2 == 0)
        output[id] = output[id] | output[id + 1];

    __syncthreads();

    if(id % 8 == 0) {
        output[id + 1] = output[id + 2];
        output[id + 2] = output[id + 4];
        output[id + 3] = output[id + 6];

        printf("S-BOX: %02X%02X%02X%02X\n", output[id + 0], output[id + 1], output[id + 2], output[id + 3]);
    }

//    printf("SBOX: id = %d: %02X -> shamt %d row: %d, col %d, res: %x, out %02x\n", id, test, (7 - modId)*6, row, col, result, output[id]);
}

__global__ void cuPPerm(byte * input, byte * output)
{
    int id = get_idx();

    output[id] = doShifting(Pf, input, id%4);

    printf("P: id = %d: %02x\n", id, output[id]);
}

__global__ void cuCombine(byte * L, byte * R, byte * output)
{
    int id = get_idx();

    if(id % 8 < 4) output[id] = L[id%4];
    else output[id] = R[id%4];

    printf("output: id = %d: %02x\n", id, output[id]);
}

int main(int argc, char **argv)
{

    std::cout << "Testing" << std::endl;

    byte * inputKey;
    byte * output;
    byte * after_shift;
    byte * round_keys;
    cudaMalloc((void**)&inputKey, sizeof(byte)*8);
    cudaMalloc((void**)&output, sizeof(byte)*7);
    cudaMalloc((void**)&after_shift, sizeof(byte)*16*7);
    cudaMalloc((void**)&round_keys, sizeof(byte)*16*6);

    byte * holding1;
    byte * holding2;
    byte * holding3;
    cudaMalloc((void**)&holding1, sizeof(byte)*8);
    cudaMalloc((void**)&holding2, sizeof(byte)*8);
    cudaMalloc((void**)&holding3, sizeof(byte)*8);


    byte key[8] = {0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10};

    byte plaintext[] = {0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10};

    cudaMemcpy(inputKey, &key, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(holding1, &plaintext, sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Calculate all keys
    cuPC1<<<1,7>>>(inputKey, output);
    cuLeftCircShifts<<<1,16>>>(output, after_shift);
    cuPC2<<<1,96>>>(after_shift, round_keys); // 96 = 16*6

    cuIP<<<1,8>>>(holding1, holding2);
    byte * L0 = holding2;
    byte * R0 = &holding2[4];

    cuEPerm<<<1,6>>>(R0, holding3);
    cuXOR<<<1,6>>>(holding3, round_keys, holding1);

    cuSBoxes<<<1,8>>>(holding1, holding3);
    cuPPerm<<<1,4>>>(holding3, holding1);

    cuXOR<<<1,4>>>(holding1, L0, holding3);
    cuCombine<<<1, 8>>>(R0, holding3, holding1);



    cudaDeviceSynchronize();

    byte result[16*6];
    cudaMemcpy(result, round_keys, 16*6, cudaMemcpyDeviceToHost);

    //    printf("Output: ");
    //    for(int i = 0; i < 16; i++) {
    //        printf("Key %d:\n", i+1);

    //        for(int j = i*6; j < i*6 + 6; j++)
    //            printf("%02X",result[j]);

    //        printf("\n");
    //    }

    //    printf("\n");

    return 0;
}
