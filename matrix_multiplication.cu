#include<stdio.h>
#include <cuda_runtime.h>

// lets write a GPU kernel to perform a 2D matrix multiplication

__global__ void matmul(int* a, int*b, int inner_dimension, int matrix_a_num_rows, int matrix_b_num_cols, int* result){
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    /*
    Okay i just wanna mention that its surprisingly hard to determine which matrix A row and matrix B column 
        each thread is responsible for multiplying. You'd have to somehow map a 1D threadID (0, 1, 2, ...) to a 2D (row, col) pair.
    This exact problem is why GPU thread blocks have 3 dimensions of threads. ThreadID's are now (x, y, z) coordinates, 
        making it MUCH easier to determine which data subset each thread is responsible for processing in multi-dimensional problems
    */
    if (row < matrix_a_num_rows && col < matrix_b_num_cols){
        int sum = 0;
        for (int i = 0; i < inner_dimension; i++){
            int matrix_a_index = row * inner_dimension + i; // row index * row_size (aka inner_dimension) gets you to the beginning of the target row. Add i to iterate through target row
            int matrix_b_index = col + matrix_b_num_cols * i; // row_size * i gets you to the beginning of the target row. Adding col gets you to the target column
            sum += a[matrix_a_index] * b[matrix_b_index];
        }
        result[(row * matrix_b_num_cols) + col] = sum;
    }
}
int main(){

    //initialize matrix A and B
    int num_rows_a = 2;
    int num_columns_a = 3;
    int num_rows_b = 3;
    int num_columns_b = 5;

    int *a = new int[num_rows_a * num_columns_a];
    int *b = new int[num_rows_b * num_columns_b];
    int *result = new int[num_rows_a * num_columns_b];

    for (int i = 0; i < num_rows_a * num_columns_a; i++){
        a[i] = i+1;
    }
    for (int i = 0; i < num_rows_b * num_columns_b; i++){
        b[i] = i+1;
    }
    // move matrixes to GPU memory
    int * gpu_a;
    int * gpu_b;
    int *gpu_result;

    cudaMalloc((void **)&gpu_a, num_rows_a * num_columns_a * sizeof(int));
    cudaMalloc((void **)&gpu_b, num_rows_b * num_columns_b * sizeof(int));
    cudaMalloc((void**)&gpu_result, sizeof(int) * num_rows_a * num_columns_b);

    cudaMemcpy(gpu_a, a, num_rows_a * num_columns_a * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, num_rows_b * num_columns_b * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(10, 10);
    // We want at least 1 thread for each element in the result matrix. This math calculates how many blocks are needed, but we round up
    dim3 gridSize((num_columns_b + blockSize.x - 1) / blockSize.x, (num_rows_a + blockSize.y - 1) / blockSize.y);
    // Start the kernel
    matmul<<<gridSize, blockSize>>>(gpu_a,gpu_b,num_columns_a, num_rows_a,num_columns_b,gpu_result); 
    // wait for all threads to finish and then copy result matrix back to host memory
    cudaDeviceSynchronize();
    cudaMemcpy(result, gpu_result, sizeof(int) * num_rows_a * num_columns_b, cudaMemcpyDeviceToHost);

    // Print the result matrix
    printf("Result Matrix (A * B):\n");
    for (int i = 0; i < num_rows_a; i++) {
        for (int j = 0; j < num_columns_b; j++) {
            printf("%d ", result[i * num_columns_b + j]);
        }
        printf("\n");
    }
    return 0;
}