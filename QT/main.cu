// System includes
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <stdint.h>
#include <string>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <math_constants.h>

// User includes
#include <definitions.h>

#define MAXBLOCKSIZE 960 // divisible by: 4, 6, 8 (set sizes) and 32 (warp size)

// MACROS
#define CEIL(x,y) (((x) + (y) - 1) / (y))

byte * holding1;
byte * R;
byte * L;
byte * holding2;
byte * E_output;
byte * Key_XOR_Output;
byte * S_Box_Output;
byte * P_Perm_Output;
byte * P_XOR_Output;
byte * inputKey;
byte * output;
byte * after_shift;
byte * round_keys;

size_t numSets;

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
    int shamt = total_key_shift_sizes[id];

    uint64_t upper = 0;
    uint64_t lower = 0;

    for(int i = 0; i <= 3; i++) {
        uint64_t temp = 0;
        temp += input[i];

        temp = temp << (3-i)*8;
        upper |= temp;
    }

    upper = upper >> 4;

    for(int i = 3; i <= 6; i++) {
        uint64_t temp = 0;
        temp |= input[i];
        temp = temp << (6 - i)*8;
        lower |= temp;
    }

    // mask out
    lower &= 0x0FFFFFFF;

    int numbits = 28; //7 nibbles
    uint64_t mask = upper >> numbits - shamt;
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

    byte temp = doShifting(PC_2, &input[round*7], id%6);
    //    if(id == 7) printf("Res: %0x\n", temp);
    round_keys[id] = temp;
}

// 64 -> 64
__global__ void cuIP(byte * input, byte * output, int numItems)
{
    int id = get_idx();
    if(id < numItems * 8)
    {
        int offset = (id/8) * 8;
        output[id] = doShifting(IP, input + offset, id%8);
    }
    //    printf("IP: id = %d: %02x\t%02x\n", id, input[id], output[id]);
}

// 32(64) -> 48
__global__ void cuEPerm(byte * input, byte * output, int numItems)
{
    int id = get_idx();
    if(id < numItems * 4)
    {
        int offset = (id/6) * 4;
        output[id] = doShifting(Ex, input + offset, id%6);
    }
    //    printf("Ex: id = %d: %02x, input %02x\n", id, output[id], input[id]);
}

// 48 -> 48
__global__ void cuFixedXOR(byte * input, byte * key, byte * output, int numItems)
{
    int id = get_idx();

    if(id < numItems * 6)
    {
        output[id] = input[id] ^ key[id%6]; // fixed size for stuff
    }
    //    printf("XOR: id = %d: in %02X, Key %02X, out %02x\n", id, input[id], key[id%fixedSize], output[id]);
}

// usually 32 -> 32
__global__ void cuXOR(byte * input, byte * operand, byte * output, int numItems)
{
    int id = get_idx();
    if(id < numItems * 4)
    {
        output[id] = input[id] ^ operand[id];
    }
    //    printf("XOR: id = %d: in %02X, Key %02X, out %02x\n", id, input[id], key[id], output[id]);
}

// 48 -> 32
__global__ void cuSBoxes(byte * input, byte * output, byte * temp, int numItems)
{
    int id = get_idx();

    if(id >= numItems * 8)
        return;

    output[id] = 0;

    int off = (id/8) * 6;

    uint64_t value = 0;
    for(int i = 0; i < 6; i++) {
        value |= input[i + off];
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

    temp[id] = 0;
    temp[id] |= result << 4*(1 - (id % 2));
    __syncthreads();

    if(id % 2 == 0)
        temp[id] = temp[id] | temp[id + 1];

    __syncthreads();

    if(id % 8 == 0) {
        output[id/2] = temp[id];
        output[id/2 + 1] = temp[id + 2];
        output[id/2 + 2] = temp[id + 4];
        output[id/2 + 3] = temp[id + 6];

        //        printf("S-BOX: %02X%02X%02X%02X\n", output[id + 0], output[id + 1], output[id + 2], output[id + 3]);
    }

    //    printf("SBOX: id = %d; input = %02X, output = %02X\n", id, input[id], output[id]);

    //    printf("SBOX: id = %d: %02X -> shamt %d row: %d, col %d, res: %x, out %02x\n", id, test, (7 - modId)*6, row, col, result, output[id]);
}

__global__ void cuPPerm(byte * input, byte * output, int numItems)
{
    int id = get_idx();
    if(id < numItems*4)
    {
        int offset = (id/4) * 4;
        output[id] = doShifting(Pf, input + offset, id%4);
    }
    //    printf("P: id = %d: in: %02X, out: %02x\n", id, input[id], output[id]);
}

__global__ void cuCombine(byte * L, byte * R, byte * output, int numItems)
{
    int id = get_idx();

    if(id < numItems*8)
    {
        int offset = (id/8) * 4;
        if(id % 8 < 4)
            output[id] = L[id%4 + offset];
        else
            output[id] = R[id%4 + offset];
    }
    //    printf("output: id = %d: %02x\n", id, output[id]);
}

__global__ void cuDeinterlace(byte * input, byte * L, byte * R, int numItems)
{
    int id = get_idx();
    if(id < numItems*8)
    {
        int offset = id/8 * 4;
        if(id % 8 < 4)
            L[id%4 + offset] = input[id];
        else
            R[id%4 + offset] = input[id];
    }
    //    printf("Deinterlace: id = %d: %02x; L %02X, R %02X\n", id, input[id], L[id], R[id]);
}

__global__ void cuIPInv(byte * input, byte * output, int numItems)
{
    int id = get_idx();
    if(id < numItems*8)
    {
        int offset = (id/8) * 8;
        output[id] = doShifting(IP_Inv, input + offset, id%8);
    }
    //    printf("IP inv: id = %d: %02x\n", id, output[id]);
}


void DES_Encrypt(byte *key, byte * input, byte * output);
void DES_Decrypt(byte *key, byte * input, byte * output);

int main(int argc, char **argv)
{

    std::ifstream inFile(argv[1], std::ios::binary | std::ios::ate);
    size_t numBytes = inFile.tellg();

    inFile.close();
    inFile.open(argv[1], std::ios::in);

    byte * plaintext = new byte[numBytes];

    inFile.read((char *)plaintext, numBytes);

    numSets = CEIL(strlen((char * )plaintext), 8);
    //    printf("numSets: %d\n", numSets);

    clock_t start, finish;
    start = clock();

    cudaMalloc((void**)&inputKey, sizeof(byte)*8*3);
    cudaMalloc((void**)&inputKey, sizeof(byte)*8*3);
    cudaMalloc((void**)&output, sizeof(byte)*7);
    cudaMalloc((void**)&after_shift, sizeof(byte)*16*7);
    cudaMalloc((void**)&round_keys, sizeof(byte)*16*6*3);

    cudaMalloc((void**)&holding1, sizeof(byte)*8 * numSets);
    cudaMalloc((void**)&L, sizeof(byte)*4 * numSets);
    cudaMalloc((void**)&R, sizeof(byte)*4 * numSets);
    cudaMalloc((void**)&holding2, sizeof(byte)*8 * numSets);
    cudaMalloc((void**)&E_output, sizeof(byte)*6 * numSets);
    cudaMalloc((void**)&Key_XOR_Output, sizeof(byte)*6 * numSets);
    cudaMalloc((void**)&S_Box_Output, sizeof(byte)*4 * numSets);
    cudaMalloc((void**)&P_Perm_Output, sizeof(byte)*4 * numSets);
    cudaMalloc((void**)&P_XOR_Output, sizeof(byte)*4 * numSets);

    //    byte key[] = { 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10,
    //                   0x01, 0x01, 0xFF, 0xFF, 0xDD, 0xDD, 0xAA, 0xAA,
    //                   0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF };

    byte key[] = "ABCDEFGHIJKLMNOPQRSTUVW";

    //    byte plaintext[] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};
    //    byte plaintext[] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
    //                         0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10 };



    cudaMemcpy(inputKey, &key, sizeof(uint64_t)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(holding1, plaintext, 8 * numSets, cudaMemcpyHostToDevice);

    // Calculate all keys
    // DES 1
    cuPC1<<<1,7>>>(inputKey, output);
    cuLeftCircShifts<<<1,16>>>(output, after_shift);
    cuPC2<<<1,96>>>(after_shift, round_keys); // 96 = 16*6

    // DES 2
    cuPC1<<<1,7>>>(inputKey+sizeof(uint64_t), output);
    cuLeftCircShifts<<<1,16>>>(output, after_shift);
    cuPC2<<<1,96>>>(after_shift, round_keys+(96)); // 96 = 16*6

    // DES 3
    cuPC1<<<1,7>>>(inputKey+2*sizeof(uint64_t), output);
    cuLeftCircShifts<<<1,16>>>(output, after_shift);
    cuPC2<<<1,96>>>(after_shift, round_keys+2*(96)); // 96 = 16*6

    byte hostKeys[96*3];
    cudaMemcpy(hostKeys, round_keys, sizeof(byte)*16*6*3, cudaMemcpyDeviceToHost);

    //    for(int i = 0; i < 16*3; i++)
    //    {
    //        if((i+1)%16==0){
    //            printf("Round %d key:\n", i+1);
    //            for(int j = 0; j < 6; j++)
    //            {
    //                printf("%02X ", hostKeys[i*6 + j]);
    //            }
    //            printf("\n");
    //        }
    //    }

    DES_Encrypt(round_keys, holding1, holding2);
    DES_Decrypt(round_keys+96, holding2, holding1);
    DES_Encrypt(round_keys+2*96, holding1, holding2);

    cudaDeviceSynchronize();
    finish = clock();
    double time_taken = (double)(finish - start)/(double)CLOCKS_PER_SEC;

    printf("Total time: %lf seconds\n", time_taken);

    //    byte output[8 * numSets];
    //    cudaMemcpy(&output, holding2, sizeof(byte)*8 * numSets, cudaMemcpyDeviceToHost);

    //    printf("Encrypted Result:\n");
    //    for(int j = 0; j < 8 * numSets; j++)
    //    {
    ////        printf("%02X ", output[j]);
    //    }
    //    printf("\n");
    ////    printf("%s\n", output);

    //    DES_Decrypt(round_keys+2*96, holding2, holding1);
    //    DES_Encrypt(round_keys+96, holding1, holding2);
    //    DES_Decrypt(round_keys, holding2, holding1);

    //    cudaMemcpy(&output, holding1, sizeof(byte)*8 * numSets, cudaMemcpyDeviceToHost);

    //    printf("\n\nDecrypted Result:\n");
    //    for(int j = 0; j < 8 * numSets; j++)
    //    {
    ////       printf("%02X ", output[j]);
    //    }
    //    printf("\n");
    ////    printf("%s\n", output);

    return 0;
}

void DES_Encrypt(byte *key, byte * input, byte * output)
{
    dim3 numThreads(MAXBLOCKSIZE, 1);

    int temp = CEIL(numSets*4, MAXBLOCKSIZE);
    dim3 numBlocks4(temp, 1);
    temp = CEIL(numSets*6, MAXBLOCKSIZE);
    dim3 numBlocks6(temp, 1);
    temp = CEIL(numSets*8, MAXBLOCKSIZE);
    dim3 numBlocks8(temp, 1);

    // Encrypt
    cuIP<<<numBlocks8, numThreads>>>(input, output, numSets);
    cuDeinterlace<<<numBlocks8, numThreads>>>(output, L, R, numSets); // put them in backwards at first!

    for(int i = 0; i < 16; i++)
    {
        //                printf("\n\nRound %d\n", i+1);
        // Expansion
        cuEPerm<<<numBlocks6, numThreads>>>(R, E_output, numSets);
        // XOR with round key
        cuFixedXOR<<<numBlocks6, numThreads>>>(E_output, &key[i*6], Key_XOR_Output, numSets);

        // S-Boxes
        cuSBoxes<<<numBlocks8, numThreads>>>(Key_XOR_Output, S_Box_Output, output, numSets);
        // P Perm
        cuPPerm<<<numBlocks4, numThreads>>>(S_Box_Output, P_Perm_Output, numSets);
        // XOR with L-1
        cuXOR<<<numBlocks4, numThreads>>>(P_Perm_Output, L, P_XOR_Output, numSets);
        // Copy L to R
        cudaMemcpy(L, R, sizeof(byte)*4 * numSets, cudaMemcpyDeviceToDevice);
        cudaMemcpy(R, P_XOR_Output, sizeof(byte)*4 * numSets, cudaMemcpyDeviceToDevice);

        //                cudaDeviceSynchronize();
        //        byte l[4];
        //        byte r[4];
        //        cudaMemcpy(l, L, 4, cudaMemcpyDeviceToHost);
        //        cudaMemcpy(r, R, 4, cudaMemcpyDeviceToHost);

        //        for(int ii = 0; ii < 4; ii++) printf("%02x ", l[ii]);
        //        for(int ii = 0; ii < 4; ii++) printf("%02x ", r[ii]);
        //        printf("\n");

    }

    // swap 32 bits
    cudaMemcpy(P_XOR_Output, R, sizeof(byte)*4 * numSets, cudaMemcpyDeviceToDevice);
    cudaMemcpy(R, L, sizeof(byte)*4 * numSets, cudaMemcpyDeviceToDevice);
    cudaMemcpy(L, P_XOR_Output, sizeof(byte)*4 * numSets, cudaMemcpyDeviceToDevice);

    cuCombine<<<numBlocks8, numThreads>>>(L, R, input, numSets);

    cuIPInv<<<numBlocks8, numThreads>>>(input, output, numSets);
}

void DES_Decrypt(byte *key, byte * input, byte * output)
{
    dim3 numThreads(MAXBLOCKSIZE, 1);

    int temp = CEIL(numSets*4, MAXBLOCKSIZE);
    dim3 numBlocks4(temp, 1);
    temp = CEIL(numSets*6, MAXBLOCKSIZE);
    dim3 numBlocks6(temp, 1);
    temp = CEIL(numSets*8, MAXBLOCKSIZE);
    dim3 numBlocks8(temp, 1);

    // Decrypt
    cuIP<<<numBlocks8, numThreads>>>(input, output, numSets);
    cuDeinterlace<<<numBlocks8, numThreads>>>(output, L, R, numSets); // put them in backwards at first!

    for(int i = 15; i >= 0; i--)
    {
        //         printf("\n\nRound %d\n", i+1);
        // Expansion
        cuEPerm<<<numBlocks6, numThreads>>>(R, E_output, numSets);
        // XOR with round key
        cuFixedXOR<<<numBlocks6, numThreads>>>(E_output, &key[i*6], Key_XOR_Output, numSets);

        // S-Boxes
        cuSBoxes<<<numBlocks8, numThreads>>>(Key_XOR_Output, S_Box_Output, output, numSets);
        // P Perm
        cuPPerm<<<numBlocks4, numThreads>>>(S_Box_Output, P_Perm_Output, numSets);
        // XOR with L-1
        cuXOR<<<numBlocks4, numThreads>>>(P_Perm_Output, L, P_XOR_Output, numSets);
        // Copy L to R
        cudaMemcpy(L, R, sizeof(byte)*4 * numSets, cudaMemcpyDeviceToDevice);
        cudaMemcpy(R, P_XOR_Output, sizeof(byte)*4 * numSets, cudaMemcpyDeviceToDevice);

        //        cudaDeviceSynchronize();
        //        byte l[4];
        //        byte r[4];
        //        cudaMemcpy(l, L, 4, cudaMemcpyDeviceToHost);
        //        cudaMemcpy(r, R, 4, cudaMemcpyDeviceToHost);

        //        for(int ii = 0; ii < 4; ii++) printf("%02x ", l[ii]);
        //        for(int ii = 0; ii < 4; ii++) printf("%02x ", r[ii]);
        //        printf("\n");

    }

    // swap 32 bits
    cudaMemcpy(P_XOR_Output, R, sizeof(byte)*4 * numSets, cudaMemcpyDeviceToDevice);
    cudaMemcpy(R, L, sizeof(byte)*4 * numSets, cudaMemcpyDeviceToDevice);
    cudaMemcpy(L, P_XOR_Output, sizeof(byte)*4 * numSets, cudaMemcpyDeviceToDevice);

    cuCombine<<<numBlocks8, numThreads>>>(L, R, input, numSets);

    cuIPInv<<<numBlocks8, numThreads>>>(input, output, numSets);
}
