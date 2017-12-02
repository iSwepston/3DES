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

byte * IN_LOOP1;
byte * IN_LOOP2;
byte * IN_LOOP3;
size_t numBlocksToEncrypt;

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
    if(id < numItems * 6)
    {
        int offset = (id/6) * 4;
        output[id] = doShifting(Ex, input + offset, id%6);
//        printf("Ex: id = %d: %02x, input %02x\n", id, output[id], input[id]);
    }
}

// 48 -> 48
__global__ void cuFixedXOR(byte * input, byte * key, byte * output, int numItems)
{
    int id = get_idx();

    if(id < numItems * 6)
    {
        output[id] = input[id] ^ key[id%6]; // fixed size for stuff
//        printf("XOR: id = %d: in %02X, Key %02X, out %02x\n", id, input[id], key[id%6], output[id]);
    }
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
        printf("IP inv: id = %d: %02x\n", id, output[id]);
    }
}


void DES_Encrypt(byte *key, byte * input, byte * output);
void DES_Decrypt(byte *key, byte * input, byte * output);
enum ActionType {
    DECRYPT,
    ENCRYPT
};

#define ARG_GENERATE_KEY "-g"
#define ARG_ENCRYPT "-e"
#define ARG_DECRYPT "-d"
#define ARG_HELP "-h"
#define KEY_BYTES 8*3
int main(int argc, char **argv)
{
    size_t availableBytes;
    size_t total;
    cudaMemGetInfo(&availableBytes, &total);
    printf("Total Memory: %lf\nFree Memory: %lf\n", ((double)total )/(1 << 20), ((double)availableBytes )/(1 << 20));

    size_t maxBytesPerEncrypt = 8+4+4+8+6+6+4+4+4+4; // based on space in future bits
    maxBytesPerEncrypt = (availableBytes / maxBytesPerEncrypt);
    printf("Max encrypt count = %lf\n", ((double)maxBytesPerEncrypt )/(1 << 20));

    ActionType work;
    if (argc < 2 ||
            strcmp(argv[1], ARG_HELP) == 0) {

        printf("Usage:\n");
        printf("Generate Key: -g: <output file>\n");
        printf("Encrypt: -e <key_file> <input_P> <output_C>:\n");
        printf("Decrypt: -d <key_file> <input_C> <output_P>\n");
        return 1;
    }

    if (strcmp(argv[1], ARG_GENERATE_KEY) == 0) { // Generate key file
        if (argc != 3) {
            printf("Invalid # of parameter specified.");
            return 1;
        }

        std::ofstream key_file(argv[2], std::ios::binary);
        if (!key_file.is_open()) {
            printf("Could not open file to write key.");
            return 1;
        }

        // rand generator
        uint32_t iseed = (uint32_t)time(NULL);
        srand (iseed);

        // generate key
        byte* des_key = new byte[KEY_BYTES];
        for(int i = 0; i < KEY_BYTES; i++)
            des_key[i] = rand() % 255;

        // write out a key
        key_file.write((char *)des_key, KEY_BYTES);

        delete des_key;
        key_file.close();
        return 0;
    } else if(strcmp(argv[1], ARG_ENCRYPT) == 0) {
        work = ENCRYPT;
        if (argc != 5) {
            printf("Invalid # of parameter specified.");
            return 1;
        }
    } else if(strcmp(argv[1], ARG_DECRYPT) == 0) {
        work = DECRYPT;
        if (argc != 5) {
            printf("Invalid # of parameter specified.");
            return 1;
        }
    } else {
        return 1;
    }

    std::ifstream keyFile(argv[2], std::ios::binary);
    std::ifstream inFile(argv[3], std::ios::binary | std::ios::ate);
    std::ofstream outFile(argv[4], std::ios::binary);

    if(!keyFile.is_open()
            || !inFile.is_open()
            || !outFile.is_open()) {
        printf("One or more files not opened!\n");
        return 1;
    }

    // count input
    size_t numBytes = inFile.tellg();
    if(numBytes % 8 != 0)
        numBytes += (8 - numBytes%8); // round to 8

    inFile.seekg(0);

    // read infile
    byte * plaintext = new byte[numBytes];
    inFile.read((char *)plaintext, numBytes);

    numBlocksToEncrypt = CEIL(numBytes, 8);
    printf("Operating on %lu blocks\n", numBlocksToEncrypt);

    printf("Reading\n");
    byte *key = new byte[KEY_BYTES];
    keyFile.read((char *)key, KEY_BYTES);

//    printf("Input:\n");
//    for(int j = 0; j < numBytes; j++)
//    {
//        printf("%02X ", plaintext[j]);
//    }
//    printf("\n");

//    printf("Key:\n");
//    for(int j = 0; j < 8; j++)
//    {
//        printf("%02X ", key[j]);
//    }
//    printf("\n");

    cudaMalloc((void**)&inputKey, sizeof(byte)*8*3);
    cudaMalloc((void**)&inputKey, sizeof(byte)*8*3);
    cudaMalloc((void**)&output, sizeof(byte)*7);
    cudaMalloc((void**)&after_shift, sizeof(byte)*16*7);
    cudaMalloc((void**)&round_keys, sizeof(byte)*16*6*3);

    cudaMalloc((void**)&holding1, sizeof(byte)*8 * numBlocksToEncrypt);
    cudaMalloc((void**)&L, sizeof(byte)*4 * numBlocksToEncrypt);
    cudaMalloc((void**)&R, sizeof(byte)*4 * numBlocksToEncrypt);
    cudaMalloc((void**)&holding2, sizeof(byte)*8 * numBlocksToEncrypt);

    cudaMalloc((void**)&E_output, sizeof(byte)*6 * numBlocksToEncrypt);
    cudaMalloc((void**)&Key_XOR_Output, sizeof(byte)*6 * numBlocksToEncrypt);
    cudaMalloc((void**)&S_Box_Output, sizeof(byte)*4 * numBlocksToEncrypt);
    cudaMalloc((void**)&P_Perm_Output, sizeof(byte)*4 * numBlocksToEncrypt);
    cudaMalloc((void**)&P_XOR_Output, sizeof(byte)*4 * numBlocksToEncrypt);

    clock_t start, finish;
    start = clock();


    cudaMemcpy(inputKey, key, sizeof(uint64_t)*3, cudaMemcpyHostToDevice);
    cudaMemcpy(holding1, plaintext, 8 * numBlocksToEncrypt, cudaMemcpyHostToDevice);

    clock_t afterCopy = clock();
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


    // Do encryption or decryption
    switch (work) {
    case ENCRYPT:
        DES_Encrypt(round_keys, holding1, holding2);
        DES_Decrypt(round_keys+96, holding2, holding1);
        DES_Encrypt(round_keys+2*96, holding1, holding2);
        break;
    case DECRYPT:
        DES_Decrypt(round_keys+2*96, holding1, holding2);
        DES_Encrypt(round_keys+96, holding2, holding1);
        DES_Decrypt(round_keys, holding1, holding2);
        break;
    default:
        return -1;
    }
    clock_t afterCompute = clock();
    // timing methods
    cudaDeviceSynchronize();
    // write to output
    cudaMemcpy(plaintext, holding2, sizeof(byte)*8 * numBlocksToEncrypt, cudaMemcpyDeviceToHost);

    finish = clock();
    double Total_time_taken = (double)(finish - start)/(double)CLOCKS_PER_SEC;
    double compute_time = (double)(afterCompute - afterCopy)/(double)CLOCKS_PER_SEC;

    printf("Total time: %lf seconds\nCompute only %lf", Total_time_taken, compute_time);

    outFile.write((char *)plaintext, numBytes);

    return 0;
}

void DES_Encrypt(byte *key, byte * input, byte * output)
{
    dim3 numThreads(MAXBLOCKSIZE, 1);

    int temp = CEIL(numBlocksToEncrypt*4, MAXBLOCKSIZE);
    dim3 numBlocks4(temp, 1);
    temp = CEIL(numBlocksToEncrypt*6, MAXBLOCKSIZE);
    dim3 numBlocks6(temp, 1);
    temp = CEIL(numBlocksToEncrypt*8, MAXBLOCKSIZE);
    dim3 numBlocks8(temp, 1);

    // Encrypt
    cuIP<<<numBlocks8, numThreads>>>(input, output, numBlocksToEncrypt);
    cuDeinterlace<<<numBlocks8, numThreads>>>(output, L, R, numBlocksToEncrypt); // put them in backwards at first!

    for(int i = 0; i < 16; i++)
    {
        //                printf("\n\nRound %d\n", i+1);
        // Expansion
        cuEPerm<<<numBlocks6, numThreads>>>(R, E_output, numBlocksToEncrypt);
        // XOR with round key
        cuFixedXOR<<<numBlocks6, numThreads>>>(E_output, &key[i*6], Key_XOR_Output, numBlocksToEncrypt);

        // S-Boxes
        cuSBoxes<<<numBlocks8, numThreads>>>(Key_XOR_Output, S_Box_Output, output, numBlocksToEncrypt);
        // P Perm
        cuPPerm<<<numBlocks4, numThreads>>>(S_Box_Output, P_Perm_Output, numBlocksToEncrypt);
        // XOR with L-1
        cuXOR<<<numBlocks4, numThreads>>>(P_Perm_Output, L, P_XOR_Output, numBlocksToEncrypt);
        // Copy L to R
        cudaDeviceSynchronize();
        if(i != 15) {
            byte * temp = L;
            L = R;
            R = P_XOR_Output;
            P_XOR_Output = temp;
        } else {
            byte * temp = L;
            L = P_XOR_Output;
            P_XOR_Output = temp;
        }

                byte l[4];
                byte r[4];
                cudaMemcpy(l, L, 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(r, R, 4, cudaMemcpyDeviceToHost);

                for(int ii = 0; ii < 4; ii++) printf("%02x ", l[ii]);
                for(int ii = 0; ii < 4; ii++) printf("%02x ", r[ii]);
                printf("\n");
    }

    cuCombine<<<numBlocks8, numThreads>>>(L, R, input, numBlocksToEncrypt);

    cuIPInv<<<numBlocks8, numThreads>>>(input, output, numBlocksToEncrypt);
}

void DES_Decrypt(byte *key, byte * input, byte * output)
{
    dim3 numThreads(MAXBLOCKSIZE, 1);

    int temp = CEIL(numBlocksToEncrypt*4, MAXBLOCKSIZE);
    dim3 numBlocks4(temp, 1);
    temp = CEIL(numBlocksToEncrypt*6, MAXBLOCKSIZE);
    dim3 numBlocks6(temp, 1);
    temp = CEIL(numBlocksToEncrypt*8, MAXBLOCKSIZE);
    dim3 numBlocks8(temp, 1);

    // Decrypt
    cuIP<<<numBlocks8, numThreads>>>(input, output, numBlocksToEncrypt);
    cuDeinterlace<<<numBlocks8, numThreads>>>(output, L, R, numBlocksToEncrypt); // put them in backwards at first!

    for(int i = 15; i >= 0; i--)
    {
        //         printf("\n\nRound %d\n", i+1);
        // Expansion
        cuEPerm<<<numBlocks6, numThreads>>>(R, E_output, numBlocksToEncrypt);
        // XOR with round key
        cuFixedXOR<<<numBlocks6, numThreads>>>(E_output, &key[i*6], Key_XOR_Output, numBlocksToEncrypt);

        // S-Boxes
        cuSBoxes<<<numBlocks8, numThreads>>>(Key_XOR_Output, S_Box_Output, output, numBlocksToEncrypt);
        // P Perm
        cuPPerm<<<numBlocks4, numThreads>>>(S_Box_Output, P_Perm_Output, numBlocksToEncrypt);
        // XOR with L-1
        cuXOR<<<numBlocks4, numThreads>>>(P_Perm_Output, L, P_XOR_Output, numBlocksToEncrypt);
        // Copy L to R
        cudaDeviceSynchronize();

        if(i != 0) {
            byte * temp = L;
            L = R;
            R = P_XOR_Output;
            P_XOR_Output = temp;
        } else {
            byte * temp = L;
            L = P_XOR_Output;
            P_XOR_Output = temp;
        }
    }

    cuCombine<<<numBlocks8, numThreads>>>(L, R, input, numBlocksToEncrypt);

    cuIPInv<<<numBlocks8, numThreads>>>(input, output, numBlocksToEncrypt);
}
