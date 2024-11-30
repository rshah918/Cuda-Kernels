#include <stdio.h>
#include <cuda_runtime.h>

/*
Alright lets start with something easy and write a GPU kernel to perform a simply vector elementwise addition
*/
__global__ void vector_addition(int*a, int*b, int vector_length, int*result){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    result[index] = a[index] + b[index];
}

int main(){
    // initialize input arrays 
    int vector_length = 5;
    int *a = new int[vector_length];
    int *b = new int[vector_length];
    int *result = new int[vector_length];

    for (int i = 0; i < vector_length; i++){
        a[i] = 1;
        b[i] = 2;
    }
    // move to GPU memory
    int *gpu_a;
    int *gpu_b;
    int *gpu_result;

    cudaMalloc((void **)&gpu_a, sizeof(int) * vector_length);
    cudaMalloc((void **)&gpu_b, sizeof(int) * vector_length);
    cudaMalloc((void **)&gpu_result, sizeof(int) * vector_length);


    cudaMemcpy(gpu_a, a, sizeof(int) * vector_length, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, sizeof(int) * vector_length, cudaMemcpyHostToDevice);

    // configure thread layout
    dim3 blockSize(50);
    dim3 gridSize((vector_length + blockSize.x - 1) / blockSize.x);
    // start kernel and wait for all threads to finish
    vector_addition<<<gridSize, blockSize>>>(gpu_a, gpu_b, vector_length, gpu_result);
    cudaDeviceSynchronize();
    // copy result vector back to host device and print result. It should show a vector of all 3's (1+2=3 in case u didnt know)
    cudaMemcpy(result, gpu_result, sizeof(int) * vector_length, cudaMemcpyDeviceToHost);
    
    printf("Result Vector (A + B):\n");
    for (int i = 0; i < vector_length; i++){
        printf("%d ", result[i]);
    }
    return 0;
}