#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void example(float *x, float *y, int nfloat)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float d;
  for (int i=0; i < 1; ++i) // one iteration
  { 
    float tmp;
    tmp = x[i] / (x[i]-y[i]); // division by zero <- mufu.rcp,r0=inf[rslt],r1=input
       // FFMA in div // NaN part of div
    d = sqrt(tmp);     // 2 reg, one reg is inf, d is inf, tmp was inf         
  }

  tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    printf("dot: %f\n", d);
  }
}

int main(int argc, char **argv)
{
  int nfloat = 2;
  int nbytes = nfloat*sizeof(float); 
  float *d_a = 0;
  cudaMalloc(&d_a, nbytes);

  float *data = (float *)malloc(nbytes);
  for (int i=0; i < nfloat; ++i)
  {
    data[i] = (float)(i+1);
  }

  cudaMemcpy((void *)d_a, (void *)data, nbytes, cudaMemcpyHostToDevice);

  printf("Calling kernel\n");
  example<<<1,1>>>(d_a, d_a, nfloat);
  cudaDeviceSynchronize();
  printf("done\n");

  return 0;
}
