#include<stdio.h>
#include <cuda_runtime.h>

/*
lets just get each GPU to print Hello World. Just read about grid-strid loop, so lets implement that as well lol 
*/
__global__ void helloFromGPU(int* a, int* b, int array_size) {
    int index = threadIdx.x + blockDim.x * threadIdx.y; // Unique thread index within a block
    int stride = blockDim.x * gridDim.x;               // Correct stride for grid-stride loop

    printf("Hello World from GPU %d!\n", index);

    for (int i = index; i < array_size; i += stride) { // Iterate over elements with proper stride
        printf("%d\n", a[i] + b[i]);
    }
}

int main(){
  // initialize 4 arrays
  int array_size = 4;
  int * a = new int[array_size];
  int * b = new int[array_size];
  for (int i = 0; i < array_size; i++){
    a[i] = i;
    b[i] = i;
  }
  // move to GPU memory
  int *d_a, *d_b;
  cudaMalloc((void**)&d_a, array_size*sizeof(int));
  cudaMalloc((void**)&d_b, array_size*sizeof(int));

  cudaMemcpy(d_a, a, array_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, array_size*sizeof(int), cudaMemcpyHostToDevice);

  // execute kernel
  helloFromGPU <<<1, 10>>>(d_a,d_b, array_size);
  cudaDeviceSynchronize(); // kernel execution is non-blocking, so need to wait for all threads to finish
  return 0;
}