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
#include <cub/cub.cuh>

#include <float.h>

// MACROS
#define GET_RDTSC(lo,hi,time) {\
    __asm__ volatile("rdtsc" : "=a" (lo), "=d" (hi));\
    time = ((uint64_t)hi << 32) | lo;\
    }

#define CEIL(x,y) (((x) + (y) - 1) / (y))

//#define ERR(x,y) ((float)abs(x-y)/(float)y)*100.0

//#define CUDART_NAN_F __int_as_float(0x7fffffff)
#define NANFINITY 1.0/0.0

#define DECRATE 32

// Constants
#define ITEMSPERTHREAD 8
#define BLOCK_SIZE 128
#define BIGBLOCK 1024


#define SMALLBLOCK 32

enum momOutputStyle {
    TRUEMEDIAN,
    FORCEPRESENT
};

// global
float * devValues;
void     *d_temp_storage;

float * devTempVals[4];

float * resultVal;
int * vals;

__global__ void cuMedianOfMedians(float * data, float * resultsArray, uint64_t dataSize, momOutputStyle outputType)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Do up to one synthetic block of work.
    int thisDataSize = min((uint64_t)ITEMSPERTHREAD*BLOCK_SIZE, dataSize);

    if(id*ITEMSPERTHREAD < dataSize) {

        __shared__ float localWorkSpace[BLOCK_SIZE*ITEMSPERTHREAD];

        int idx,j,flag,val;
        int startidx;
        float temp;
        float *local;

        // figure out your personal elements
        idx = id*ITEMSPERTHREAD;
        startidx = idx % (BLOCK_SIZE*ITEMSPERTHREAD);
        local = &localWorkSpace[startidx];

        // initial loadint of shared memory
        if(startidx < dataSize) {

            // load data into shared memory
            for(j = 0; j < ITEMSPERTHREAD && (startidx + j) < thisDataSize; j++) {
                local[j] = data[idx+j];
            }
        }
        while (thisDataSize > 1) {

            __syncthreads();

            // remove uneeded threads
            if(startidx < thisDataSize) {

                int currNumItems = 0;
                for(j = 0; j < ITEMSPERTHREAD && (startidx + j) < thisDataSize; j++) {
                    if(local[j] != NANFINITY) {
                        currNumItems++;
                    }

                }

                int numItemsToSort = ITEMSPERTHREAD;
                if(startidx + ITEMSPERTHREAD > thisDataSize) {
                    numItemsToSort = thisDataSize - startidx;
                }

                if(currNumItems > 0) {

                    // bubble sort
                    do {
                        flag = 0;
                        for(j = 0; j < (numItemsToSort-1); j++) {

                            if(local[j+1] < local[j]) {
                                temp = local[j];
                                local[j] = local[j+1];
                                local[j+1] = temp;
                                flag = 1;
                            }

                        }
                    } while(flag == 1);

                    if(outputType == TRUEMEDIAN) {

                        if(currNumItems%2 != 0) {
                            temp = local[(currNumItems-1)/2];
                        } else {
                            val = (currNumItems-1)/2;
                            temp = (local[val] + local[val+1]) / 2;

                        }
                    } else { // force result to be present
                        temp = local[(currNumItems-1)/2];
                    }
                    // set your result back in place
                    localWorkSpace[id%BLOCK_SIZE] = temp;
                } else {
                    // set your result back in place
                    localWorkSpace[id%BLOCK_SIZE] = NANFINITY;
                }

                //                if(id % 5 == 0)
                //                    printf("Size %d, Thread %d, median %f\n", thisDataSize, id, localWorkSpace[id%BLOCK_SIZE]);

            }

            // new data size
            thisDataSize = CEIL(thisDataSize,ITEMSPERTHREAD);
        }

        // Result SHOULD be in the first element of shared memory
        if(id%BLOCK_SIZE == 0) {
            //            printf("Thread %d, Block %d, median %f\n", id, id/BLOCK_SIZE, localWorkSpace[0]);
            resultsArray[id/BLOCK_SIZE] = localWorkSpace[0];
        }
    }
}


__global__ void cubMedianOfMedians(float * data, float * resultsArray, uint64_t dataSize)
{
    int blockStart = blockIdx.x * blockDim.x;
    int id = threadIdx.x + blockStart;

    int idx,j;
    int startidx;
    float local[ITEMSPERTHREAD];

    // figure out your personal elements
    idx = id*ITEMSPERTHREAD;
    startidx = idx % (BLOCK_SIZE*ITEMSPERTHREAD);

    if(id*ITEMSPERTHREAD < dataSize) {
        // load data into shared memory
        for(j = 0; j < ITEMSPERTHREAD; j++) {
            if(startidx + j < dataSize)
                local[j] = data[idx+j];
            else
                local[j] = NANFINITY;
        }
    } else {
        for(j = 0; j < ITEMSPERTHREAD; j++) {
            local[j] = NANFINITY;
        }
    }

    // Specialize BlockRadixSort for a 1D block of BLOCK_SIZE threads owning ITEMSPERTHREAD floats items each
    typedef cub::BlockRadixSort<float, BLOCK_SIZE, ITEMSPERTHREAD> BlockRadixSort;

    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // Collectively sort the keys
    __syncthreads();
    BlockRadixSort(temp_storage).Sort(local);

    int location = min(BLOCK_SIZE*ITEMSPERTHREAD, (int)(dataSize - blockStart*ITEMSPERTHREAD)) / 2;

    __syncthreads();

    if(location >= startidx && location < startidx+ITEMSPERTHREAD) {
        resultsArray[id/BLOCK_SIZE] = local[location-startidx];
    }
}

__global__ void cuDecimate(float * input, float * output, size_t numItems, int decRate) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < numItems
            && id % decRate == 0) {
        output[id/decRate] = input[id];
//        printf("Output[%d] = %f\n", id/decRate, output[id/decRate]);
    }
}

__global__ void dataSplitCompact(float * input,
                                 float * less,
                                 int * lessCount,
                                 float * more,
                                 int * moreCount,
                                 float * median,
                                 long dataSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < dataSize) {
        int myIdx;
        float * myArr;
        float myVal = input[id];

        // Atomically Guess your location
        if(myVal <= median[0]) {

            myIdx = atomicAdd(lessCount, 1);
            myArr = less;
        } else {
            myIdx = atomicAdd(moreCount, 1);
            myArr = more;
        }

        // write it
        myArr[myIdx] = myVal;
    }
}

__global__ void resetVals(int * vals, int numVals) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < numVals) {
        vals[id] = 0;
    }
}

__global__ void bubbleSortKernel(float * data, unsigned long numDataPoints) {

    float local[ITEMSPERTHREAD];
    int i, flag;
    float temp;
    for(i = 0; i < numDataPoints; i++) {
        local[i] = data[i];
    }

    // bubble sort
    do {
        flag = 0;
        for(i = 0; i < numDataPoints -1; i++) {

            if(local[i+1] < local[i]) {
                temp = local[i];
                local[i] = local[i+1];
                local[i+1] = temp;
                flag = 1;
            }

        }
    } while(flag == 1);

    for(int i = 0; i < numDataPoints; i++) {
        data[i] = local[i];
    }

}


float * medianOfMedians(float * data, unsigned long numDataPoints, momOutputStyle outputStyle) {

    // setup the cuda sizes
    dim3 numThreads(BLOCK_SIZE, 1);

//    int temp = CEIL(numDataPoints, BLOCK_SIZE);
//    temp = CEIL(temp, ITEMSPERTHREAD);

    //dim3 numBlocks(temp, 1);

    //    printf("Launching: %d blocks of %d threads\n", numBlocks.x, numThreads.x);

    float * input = data;


    unsigned long dataPointsRemaining = numDataPoints;

    int rotator = 0;
    for(rotator = 0; rotator<4; rotator++) {
        if(devTempVals[rotator] != data)
            break;
    }
    float * output = devTempVals[rotator];

    while (dataPointsRemaining > 1) {
        //        printf("Calling Kernel!!! %d elements\n", dataPointsRemaining);
        int temp = CEIL(dataPointsRemaining, BLOCK_SIZE);
        temp = CEIL(temp, ITEMSPERTHREAD);
        dim3 numBlocks(temp, 1);

        cuMedianOfMedians<<<numBlocks, numThreads>>>(input,
                                                     output,
                                                     dataPointsRemaining,
                                                     outputStyle);

        input = output;

        rotator = (rotator + 1)%4;
        if(devTempVals[rotator] == data)
            rotator = (rotator + 1)%4;

        output = devTempVals[rotator];

        // adjust counter
        dataPointsRemaining = CEIL(dataPointsRemaining,BLOCK_SIZE*ITEMSPERTHREAD);
    }

    // at the end this pointer contains the results (I think)
    return input;
}

float * cubMedianOfMedians(float * data, unsigned long numDataPoints) {

    // setup the cuda sizes
    dim3 numThreads(BLOCK_SIZE, 1);

//    int temp = CEIL(numDataPoints, BLOCK_SIZE);
//    temp = CEIL(temp, ITEMSPERTHREAD);

//    dim3 numBlocks(temp, 1);

    float * input = data;


    unsigned long dataPointsRemaining = numDataPoints;

    int rotator = 0;
    for(rotator = 0; rotator<4; rotator++) {
        if(devTempVals[rotator] != data)
            break;
    }
    float * output = devTempVals[rotator];

    while (dataPointsRemaining > 1) {
        int temp = CEIL(dataPointsRemaining, BLOCK_SIZE);
        temp = CEIL(temp, ITEMSPERTHREAD);
        dim3 numBlocks(temp, 1);

        cubMedianOfMedians<<<numBlocks, numThreads>>>(input,
                                                      output,
                                                      dataPointsRemaining);

        input = output;

        rotator = (rotator + 1)%4;
        if(devTempVals[rotator] == data)
            rotator = (rotator + 1)%4;

        output = devTempVals[rotator];

        // adjust counter
        dataPointsRemaining = CEIL(dataPointsRemaining,BLOCK_SIZE*ITEMSPERTHREAD);
    }

    // at the end this pointer contains the results (I think)
    return input;
}

float findMedian(float * data, unsigned long numItems) {

    unsigned int lo, hi;
    unsigned int cpu_clock = 1800000000;
    size_t start, stop;

    GET_RDTSC(lo, hi, start);

    int range[2];
    int targetIdx = numItems/2;

    resetVals<<<1,2>>>(vals, 2);

    float * resData = cubMedianOfMedians(data, numItems);
    // copy the output
    cudaMemcpy(resultVal, resData, sizeof(float), cudaMemcpyDeviceToDevice);

    // setup the cuda sizes
    dim3 numThreads(BLOCK_SIZE, 1);

    int temp = CEIL(numItems, BLOCK_SIZE);
    dim3 numBlocks(temp, 1);

    int rotator = 0;
    float * input = data;
    float * left = devTempVals[rotator];
    float * right = devTempVals[rotator+1];

    dataSplitCompact<<<numBlocks,numThreads>>>(data, left, &vals[0], right, &vals[1], resultVal, numItems);

    int lessCnt;
    //    int moreCnt;
    int remainingItems = numItems;
    cudaMemcpy(&lessCnt, &vals[0], sizeof(int), cudaMemcpyDeviceToHost);

    if(lessCnt <= targetIdx) {

        range[0] = lessCnt;
        range[1] = numItems - 1;

        rotator = (rotator+2)%3;

        // choose "0" keep "1" and "2"
        input = right;
        left = devTempVals[rotator];
        right = devTempVals[(rotator+1)%3];

        remainingItems = remainingItems-lessCnt;
    } else {

        range[0] = 0;
        range[1] = lessCnt-1;

        rotator = (rotator+1)%3;

        // choose "0" keep "1" and "2"
        input = left;
        left = devTempVals[rotator];
        right = devTempVals[(rotator+1)%3];

        remainingItems = lessCnt;
    }

    while(remainingItems > ITEMSPERTHREAD*BLOCK_SIZE) {

        resData = cubMedianOfMedians(input, remainingItems);

        // copy the output
        cudaMemcpy(resultVal, resData, sizeof(float), cudaMemcpyDeviceToDevice);

        resetVals<<<1,2>>>(vals, 2);

        numBlocks.x = CEIL(remainingItems, BLOCK_SIZE);

        dataSplitCompact<<<numBlocks,numThreads>>>(input, left, &vals[0], right, &vals[1], resultVal, remainingItems);

        cudaMemcpy(&lessCnt, &vals[0], sizeof(int), cudaMemcpyDeviceToHost);

        if(targetIdx <= (range[0] + lessCnt - 1)) { // choose the "left"
            // range adjustment
            range[1] = range[0] + lessCnt - 1;
            remainingItems = lessCnt;

            rotator = (rotator+1)%3;

            // choose "0" keep "1" and "2"
            input = left;
            left = devTempVals[rotator];
            right = devTempVals[(rotator+1)%3];

        } else { // choose the "right"
            // range adjustment
            range[0] = range[0] + lessCnt;
            remainingItems = remainingItems - lessCnt;

            rotator = (rotator+2)%3;

            // choose "0" keep "1" and "2"
            input = right;
            left = devTempVals[rotator];
            right = devTempVals[(rotator+1)%3];
        }
    }

    size_t tempBytes;
    cub::DeviceRadixSort::SortKeys(NULL, tempBytes, input, left, remainingItems);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, tempBytes, input, left, remainingItems);

    cudaDeviceSynchronize();
    GET_RDTSC(lo, hi, stop);

    float median;
    int loi = targetIdx - range[0];

    cudaMemcpy(&median, &left[loi], sizeof(float), cudaMemcpyDeviceToHost);

    printf("Median is %f\nEx Time %lf ms\n\n", median, ((double)(stop - start))/cpu_clock * 1000);
    return median;
}


/// New Method:
///
/// Just "Swap" the data
///
__global__ void dataSplit(float * input,
                          float * less,
                          float * more,
                          float * median,
                          long dataSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    float myVal = input[id];
    float * myArr;
    float * theirArr;

    if(id < dataSize) {
        // Swapem
        if(myVal <= median[0]) {
            myArr = less;
            theirArr = more;
        } else {
            myArr = more;
            theirArr = less;
        }
        // write it
        myArr[id] = myVal;
        theirArr[id] = NANFINITY;
    }
}


__global__ void reduceCountKernel(float * input, int * outputCount, long dataSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ int values[BIGBLOCK];

    values[id] = 0;

    int lower = dataSize/blockDim.x * id;
    int upper = dataSize/blockDim.x * (id+1);

    if(id == blockDim.x-1) {
        upper += dataSize % blockDim.x;
    }

    if(id < dataSize) {

        for(int i = lower; i < upper && i < dataSize; i++) {
            if(input[i] != NANFINITY) {
                values[id]++;
            }
        }
    }

    //    printf("ID: %d, Count %d\n", id, values[id]);
    __syncthreads();

    int localCount = 0;
    if(id == 0) {
        for(int i = 0; i < blockDim.x; i++) {
            localCount += values[i];
        }

        *outputCount = localCount;
    }


}

__global__ void compactKernel(float * input,
                              float * output,
                              int * counter,
                              long dataSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < dataSize) {
        int myIdx;
        float myVal = input[id];

        if(myVal != NANFINITY) {
            myIdx = atomicAdd(counter, 1);

            // write it
            output[myIdx] = myVal;
        }
    }
}

// select a random value! SO SLOW
__global__ void randValKernel(float * input, float * outputVal, size_t seed, long dataSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float sharedValues[SMALLBLOCK];

    int lower = dataSize/blockDim.x * id;
    int upper = dataSize/blockDim.x * (id+1);

    if(lower < dataSize) {

        int numItems = upper - lower;

        curandState_t state;
        curand_init(seed, id, 0, &state);
        int offset = curand(&state) % numItems;
        float myVal;

        for(int i = 0; i < numItems; i++) {

            myVal = input[lower + ((i+offset)%numItems)];

            if(abs(myVal) != NANFINITY)
                break;
        }

        sharedValues[id] = myVal;

        __syncthreads();

        if(id == 0) {

            offset = curand(&state) % SMALLBLOCK;

            for(int i = 0; i < SMALLBLOCK; i++) {

                myVal = sharedValues[((i+offset)%SMALLBLOCK)];

                if(myVal != NANFINITY)
                    break;
            }

            outputVal[0] = myVal;
        }
    }
}

__global__ void randValPrescanSetupKernel(float * outputVals, long dataSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < dataSize) {
        outputVals[id] = NANFINITY;
    }

}

// prescan for rand val selection
__global__ void randValPrescanKernel(float * input, float * outputVals, size_t randoffset, long dataSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < dataSize) {
        float myVal = input[(id+randoffset)%dataSize];

        //        printf("Output TID: %d, Value %f\n", id, outputVals[id/BIGBLOCK]);

        if(myVal != NANFINITY) {
            outputVals[id/BIGBLOCK] = myVal;
        }

        //        printf("Post-output TID: %d, Value %f\n", id, outputVals[id/BIGBLOCK]);
    }

}

// prescan for rand val selection
__global__ void randValQuickKernel(float * input, float * outputVal, size_t randoffset, long dataSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id == 0)
    {
        printf("Rotate by: %ld\n", randoffset%dataSize);
    }

    if(id < dataSize) {
        float myVal = input[(id+randoffset)%dataSize];

        if(myVal != NANFINITY) {
            outputVal[0] = myVal;
        }
    }

}

// prescan for rand val selection
__global__ void randValQuickerKernel(float * input, float * outputVal, size_t randoffset, long dataSize) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id*ITEMSPERTHREAD < dataSize) {

        int lower = ITEMSPERTHREAD * id;
        int upper = ITEMSPERTHREAD * (id+1);

        if(id == blockDim.x-1) {
            upper += dataSize % ITEMSPERTHREAD;
        }

        float myVal;
        for(int i = 0; i < ITEMSPERTHREAD; i++) {
            int idx = lower + (i + (randoffset%dataSize))%ITEMSPERTHREAD;

            myVal = input[(idx + (randoffset*randoffset))%dataSize];

            if(myVal != NANFINITY) {
                outputVal[0] = myVal;
                break;
            }
        }
    }


}

float medianNewWay(float * data, unsigned long numItems) {

    float * a;
    float * b;

    unsigned int lo, hi;
    size_t start, seed, stop;

    cudaMallocHost((void**)&a, sizeof(float)*numItems);
    cudaMallocHost((void**)&b, sizeof(float)*numItems);

    int range[2];
    int targetIdx = numItems/2;

    int remainingItems = numItems;
    range[0] = 0;
    range[1] = numItems-1;

    int rotator = 0;
    float * input = data;
    float * left = devTempVals[rotator];
    float * right = devTempVals[rotator+1];

    // setup the cuda sizes
    dim3 numThreads(BLOCK_SIZE, 1);

    int temp = CEIL(numItems, BLOCK_SIZE);
    dim3 numBlocks(temp, 1);

    float * resData = &devValues[0];

    int count = 0;
    GET_RDTSC(lo, hi, start);

    while(remainingItems > ITEMSPERTHREAD) {
        count++;
        //        printf("Remaining Items: %d\n", remainingItems);
        //        while(1) {
        GET_RDTSC(lo, hi, seed);


        //            randValPrescanSetupKernel<<<1, BIGBLOCK>>>(devTemp1, BIGBLOCK);
        ////            cudaMemcpy(a, devTemp1, sizeof(float)*BIGBLOCK, cudaMemcpyDeviceToHost);
        ////            for(int i = 0; i < BIGBLOCK; i++)
        ////                printf("a[%d] = %f\n", i, a[i]);

        //            randValPrescanKernel<<<numBlocks, numThreads>>>(input, devTemp1, seed, numItems);

        ////            cudaMemcpy(a, devTemp1, sizeof(float)*BIGBLOCK, cudaMemcpyDeviceToHost);
        ////            for(int i = 0; i < BIGBLOCK; i++)
        ////                printf("a[%d] = %f\n", i, a[i]);

        //            randValKernel<<<1, SMALLBLOCK>>>(devTemp1, resData, seed, BIGBLOCK);

        dim3 numBlocks2(CEIL(temp,ITEMSPERTHREAD), 1);

        randValQuickerKernel<<<numBlocks2, numThreads>>>(input, resData, seed, numItems);
        cudaDeviceSynchronize();
        //        GET_RDTSC(lo,hi,stop);
        ////        printf("%lf ms\n\n\n", ((double)(stop-seed))/2500000);

        resetVals<<<1,2>>>(vals, 2);

        //        float * resData = medianOfMedians(input, numItems, FORCEPRESENT);

        //        float median;
        //        cudaMemcpy(&median, resData, sizeof(float), cudaMemcpyDeviceToHost);

        //        printf("Using Median: %f\n", median);
        //        //            usleep(1000*100);
        //        //        }
        dataSplit<<<numBlocks,numThreads>>>(input, left, right, resData, numItems);


        reduceCountKernel<<<1,BIGBLOCK>>>(left, &vals[0], numItems);
        //        reduceCountKernel<<<1,BIGBLOCK>>>(right, &vals[1], numItems);

        int lessCnt;
        cudaMemcpy(&lessCnt, vals, sizeof(int), cudaMemcpyDeviceToHost);

        //        printf("Less Count: %d\n", lessCnt);

        if(targetIdx <= (range[0] + lessCnt - 1)) { // choose the "left"
            //            printf("Choose 'Left'\n");
            // range adjustment
            range[1] = range[0] + lessCnt - 1;
            remainingItems = lessCnt;

            rotator = (rotator+1)%3;

            // choose "0" keep "1" and "2"
            input = left;
            left = devTempVals[rotator];
            right = devTempVals[(rotator+1)%3];

        } else { // choose the "right"
            //            printf("Choose 'Right'\n");
            // range adjustment
            range[0] = range[0] + lessCnt;
            remainingItems = remainingItems - lessCnt;

            rotator = (rotator+2)%3;

            // choose "0" keep "1" and "2"
            input = right;
            left = devTempVals[rotator];
            right = devTempVals[(rotator+1)%3];
        }

        //        printf("Range: [%d, %d]\n\n", range[0], range[1]);

        //        cudaMemcpy(a, input, sizeof(float)*numItems, cudaMemcpyDeviceToHost);

        //        for(int i = 0; i < numItems; i++)
        //            printf("%d: %f\n", i, a[i]);

        //        usleep(1000*200);
    }



    resetVals<<<1,2>>>(vals, 2);
    compactKernel<<<numBlocks, numThreads>>>(input,devTempVals[0],vals,numItems);
    bubbleSortKernel<<<1,1>>>(devTempVals[0],remainingItems);

    cudaDeviceSynchronize();
    GET_RDTSC(lo,hi,stop);
    printf("numcycles %d, %lf ms\n", count, ((double)(stop-start))/2500000);

    float result;
    int loi = targetIdx - range[0];

    cudaMemcpy(&result, &devTempVals[0][loi], sizeof(float), cudaMemcpyDeviceToHost);

    printf("Median New Way is: %f\n\n\n", result);

    return 1.0;
}


int main(int argc, char **argv)
{
    unsigned long numDataPoints = atoi(argv[1]);
    printf("Running for %ld datapoints\n", numDataPoints);
    int devID = 0;
    unsigned int lo, hi;
    unsigned int cpu_clock = 1800000000;
    size_t start, stop;

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    /// Allocate memory
    ///
    cudaMalloc((void**)&devValues, sizeof(float)*numDataPoints);
    cudaMalloc((void**)&devTempVals[0], sizeof(float)*numDataPoints);
    cudaMalloc((void**)&devTempVals[1], sizeof(float)*numDataPoints);
    cudaMalloc((void**)&devTempVals[2], sizeof(float)*numDataPoints);
    cudaMalloc((void**)&devTempVals[4], sizeof(float)*numDataPoints);
    cudaMalloc((void**)&resultVal, sizeof(float));
    cudaMalloc((void**)&vals, 2*sizeof(int));

    float * hVals;
    cudaMallocHost((void**)&hVals, sizeof(float)*numDataPoints);


    /// CUB Device level setup
    ///

    // Determine temporary device storage requirements
    d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, devValues, devTempVals[0], numDataPoints);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    /// Generate Random Floats for Testing
    ///
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    GET_RDTSC(lo, hi, start);
    curandSetPseudoRandomGeneratorSeed(prng, start);

    curandGenerateUniform(prng, devValues, numDataPoints);


    for(int i = 0; i < numDataPoints; i++) {
        hVals[i] = i;// (float)i-(float)(numDataPoints/2.0);
//        printf("Vals: %lf\n", hVals[i]);
    }

//    cudaMemcpy(devValues, hVals, numDataPoints*sizeof(float), cudaMemcpyHostToDevice);

    // CPU True Median
    cudaMemcpy(hVals, devValues, sizeof(float)*numDataPoints, cudaMemcpyDeviceToHost);

    GET_RDTSC(lo, hi, start);
    std::nth_element(hVals, hVals + numDataPoints/2, hVals+numDataPoints);
    GET_RDTSC(lo, hi, stop);
    printf("CPU Median: Calculation took: %lf ms\n", ((double)(stop - start))/cpu_clock * 1000);
    float trueMedian = hVals[numDataPoints/2];
    printf("CPU True Median is %f\n\n", trueMedian);

    ///
    /// Cub Full Sort Select
    ///
    cudaDeviceSynchronize();
    // Run sorting operation
    GET_RDTSC(lo, hi, start);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, devValues, devTempVals[0], numDataPoints);
    cudaDeviceSynchronize();
    GET_RDTSC(lo, hi, stop);
    printf("CUB Sorting took: %lf ms\n", ((double)(stop - start))/cpu_clock * 1000);


    float cubMed;
    cudaMemcpy(&cubMed, devTempVals[0]+numDataPoints/2, sizeof(float), cudaMemcpyDeviceToHost);

    printf("CUB Median is: %f, %% Error %f\n\n", cubMed, abs((cubMed-trueMedian)/trueMedian)*100);

    ///
    /// Cub Decimated sort/select
    ///
    // setup the cuda sizes
    dim3 numThreads(BLOCK_SIZE, 1);
    int temp = CEIL(numDataPoints, BLOCK_SIZE);
    dim3 numBlocks(temp, 1);

    cub::DeviceRadixSort::SortKeys(NULL, temp_storage_bytes, devTempVals[0], devTempVals[1], numDataPoints/DECRATE);

    cudaDeviceSynchronize();
    // Run sorting operation
    GET_RDTSC(lo, hi, start);
    cuDecimate<<<numBlocks,numThreads>>>(devValues, devTempVals[0], numDataPoints, DECRATE);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, devTempVals[0], devTempVals[1], numDataPoints/DECRATE);
    cudaDeviceSynchronize();
    GET_RDTSC(lo, hi, stop);
    printf("CUB Decimated Sorting took: %lf ms\n", ((double)(stop - start))/cpu_clock * 1000);

    float cubDecMed;
    cudaMemcpy(&cubDecMed, devTempVals[1]+numDataPoints/DECRATE/2, sizeof(float), cudaMemcpyDeviceToHost);

    printf("CUB Decimated Median is: %f, %% Error %f\n\n", cubDecMed, abs((cubDecMed-trueMedian)/trueMedian)*100);

    ///
    /// Normal Median of Medians
    ///
    cudaDeviceSynchronize();
    GET_RDTSC(lo, hi, start);
    float * result = medianOfMedians(devValues, numDataPoints, FORCEPRESENT);
    cudaDeviceSynchronize();
    GET_RDTSC(lo, hi, stop);
    printf("MOM: Calculation took: %lf ms\n", ((double)(stop - start))/cpu_clock * 1000);

    float medOfMed;
    cudaMemcpy(&medOfMed, result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Median of Medians is: %f, %% Error %f\n\n", medOfMed, abs((medOfMed-trueMedian)/trueMedian)*100);


    ///
    /// CUB Median of Medians
    ///
    cudaDeviceSynchronize();
    GET_RDTSC(lo, hi, start);
    float * result2 = cubMedianOfMedians(devValues, numDataPoints);
    cudaDeviceSynchronize();
    GET_RDTSC(lo, hi, stop);
    printf("CUBMOM: Calculation took: %lf ms\n", ((double)(stop - start))/cpu_clock * 1000);

    float medOfMed2;
    cudaMemcpy(&medOfMed2, result2, sizeof(float), cudaMemcpyDeviceToHost);

    printf("CUB Median of Medians is: %f, %% Error %f\n\n", medOfMed2, abs((medOfMed2-trueMedian)/trueMedian)*100);


//    cudaMemcpy(hVals, devValues, sizeof(float)*numDataPoints, cudaMemcpyDeviceToHost);

//    GET_RDTSC(lo, hi, start);
//    std::nth_element(hVals, hVals + numDataPoints/2, hVals+numDataPoints);
//    GET_RDTSC(lo, hi, stop);
//    printf("CPU Median: Calculation took: %lf ms\n", ((double)(stop - start))/cpu_clock * 1000);
//    printf("CPU True Median is %f\n\n", hVals[numDataPoints/2]);

    //    sleep(2);

    findMedian(devValues, numDataPoints);

    //    sleep(1);
    medianNewWay(devValues, numDataPoints);
}
