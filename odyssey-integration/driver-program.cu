#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>


// This is our device function template
// DEVICE_FUNCTION_BODY will be replaced during compilation
__device__ float dynamic_function(float x) {
    return DEVICE_FUNCTION_BODY;
}

__global__ void kernel_function(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = dynamic_function(input[idx]);
        
        // Debug print for first thread
        if (idx == 0) {
            printf("Input: %f, Output: %f\n", input[idx], output[idx]);
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s \"expression\"\n", argv[0]);
        printf("Example: %s \"x * x\"\n", argv[0]);
        return 1;
    }

    // Setup data
    const int N = 1024;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Using expression: %s\n", argv[1]);
    kernel_function<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print first few results
    printf("\nFirst 5 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("Input: %f, Output: %f\n", h_input[i], h_output[i]);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}