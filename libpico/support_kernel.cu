#include <cuda.h> 
#include <stdio.h>
#include "support_kernel.h"

__global__ 
void reduction_int32(int32_t *inbuff, int32_t *inoutbuff, int count)
{
  int index = threadIdx.x;
  printf("started kernel %d\n", index);
  if (index < count)
  {
    inoutbuff[index] = inoutbuff[index] + inbuff[index];
  }
}

void reduce_wrapper(void* inbuff, void* inoutbuff, int count) {
  reduction_int32<<<1, count>>>((int *)inbuff, (int *)inoutbuff, count);
}
