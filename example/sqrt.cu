#include <stdio.h>
#include <stdlib.h>

__global__ void some_kernel(float *x, float *y, int size)
{
  float d;
  for (int i=0; i < size; ++i)
  {
    float tmp;
    tmp = x[i] / (x[i]-x[i]);  // div by zero => INF
    d = sqrt(tmp);             // nan because of sqrt
  }
}
int main(int argc, char **argv)
{
  int n = 3;
  int nbytes = n*sizeof(float); 
  float *d_a = 0;
  cudaMalloc(&d_a, nbytes);

  float *data = (float *)malloc(nbytes);
  for (int i=0; i < n; ++i)
  {
    data[i] = (float)(i+1);
  }

  cudaMemcpy((void *)d_a, (void *)data, nbytes, cudaMemcpyHostToDevice);

  printf("Calling kernel\n");
  some_kernel<<<1,1>>>(d_a, d_a, nbytes);
  cudaDeviceSynchronize();
  printf("done\n");

  return 0;
}
