#include <stdio.h>
#include <cuda_runtime.h>

__device__ double e = 2.71828182845904523536;
// Device function to compute a math expression with one variable

__device__ float compute_expression(float x) {
    return DEVICE_FUNCTION_BODY;
}
// CUDA kernel to compute expressions
__global__ void compute_kernel(float* x, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = compute_expression(x[idx]);
    }
}

// Host function to set up and launch kernel
void launch_computation(float* d_x, float* d_result, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_result, n);
}

int main() {
    const int N = 100;  // Number of elements to process

    // Allocate host memory
    float *h_x, *h_result;
    h_x = (float*)malloc(N * sizeof(float));
    h_result = (float*)malloc(N * sizeof(float));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;  // Random values between -5 and 5
    }

    // Allocate device memory
    float *d_x, *d_result;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    // Launch computation
    launch_computation(d_x, d_result, N);

    // Copy results back to host
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    // // Print a few results for verification
    // for (int i = 0; i < 10; i++) {
    //     printf("For x = %f, Result = %f\n", h_x[i], h_result[i]);
    // }

    // Clean up
    free(h_x); free(h_result);
    cudaFree(d_x); cudaFree(d_result);

    return 0;
}