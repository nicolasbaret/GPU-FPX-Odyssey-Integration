#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Constants
__device__ const float e = 2.71828182845904523536f;
__device__ const float pi = 3.14159265358979323846f;

// This is our device function with the expression directly embedded
__device__ float dynamic_function(float x) {
    return DEVICE_FUNCTION_BODY;
}

__global__ void kernel_function(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (input[idx] <= 0) {
            output[idx] = 0.0f;  // default value for invalid inputs
            return;
        }
        
        output[idx] = dynamic_function(input[idx]);
        
        // Debug print for first thread
        if (idx == 0) {
            printf("First value - Input: %f, Output: %f\n", input[idx], output[idx]);
        }
    }
}

int main(int argc, char **argv) {
    // Setup data
    const int N = 1024;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i + 1);  // Start from 1 to avoid log(0)
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
    
    printf("Executing kernel with expression\n");
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