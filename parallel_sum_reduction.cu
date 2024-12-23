#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;
/*
Alright lets do something fancier: Parallel Sum Reduction. This is a semi-scary way to say "sum up everything in this vector".
Lets say our input is [1, 2, 3, 4, 5, 6]:

We want to do as many additions in parallel as possible. To accomplish this, we'd add each pair of numbers and
    collapse their sums into the first element

For example with input [1, 2, 3, 4, 5, 6]:
    Step 1 (stride = 1): shared_data = [3, 5, 7, 9, 11, 6]
    Step 2 (stride = 2): shared_data = [10, 14, 18, 15, 11, 0]
    Step 3 (stride = 4): shared_data = [21, 0, 0, 0, 0, 0]
    Final Result: 21
*/
__global__ void parallel_sum_reduction(int* input, int* result, int size){
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int shared_data[1024];
    if (global_thread_index < size){
       shared_data[threadIdx.x] = input[global_thread_index];
    }
    else{
        shared_data[threadIdx.x] = 0;
    }
    for (int stride = 1; stride < size; stride = stride * 2){
        if(global_thread_index < size){
            shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if(global_thread_index == 0){
        *result=shared_data[0];    
    }
}

int main(){
    //create input array
    int input_length = 6;
    int input[input_length] = {1, 2, 3, 4, 5, 6};
    // allocate gpu memory
    int *gpu_inp;
    int *gpu_result;
    cudaMalloc((void**)&gpu_inp, sizeof(int) * input_length);    
    cudaMalloc((void**)&gpu_result, sizeof(int));
    // move to gpu memory
    cudaMemcpy(gpu_inp, input, sizeof(int) * input_length, cudaMemcpyHostToDevice);
    // launch kernel
    dim3 blockSize(32);
    dim3 gridSize((input_length + blockSize.x - 1) / blockSize.x);
    parallel_sum_reduction<<<gridSize, blockSize>>>(gpu_inp, gpu_result, 6);
    // wait for all threads to finish
    cudaDeviceSynchronize();
    // copy result to host memory and print
    int result;
    cudaMemcpy(&result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum is: %d\n", result);
    return 0;
}