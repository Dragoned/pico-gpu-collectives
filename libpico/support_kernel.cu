#include "support_kernel.h"

#define MAX_THERAD 1024
#define GLOBAL_IDX (blockIdx.x * blockDim.x + threadIdx.x)

#define MAKE_KERNEL_MAX(type, name)                                                      \
  __global__ void name(type *inbuff, type *inountbuff, int n)                            \
  {                                                                                      \
    int idx = GLOBAL_IDX;                                     \
    if (idx < n)                                                                         \
      inountbuff[idx] = (inbuff[idx] > inountbuff[idx]) ? inbuff[idx] : inountbuff[idx]; \
  }

#define MAKE_KERNEL_MIN(type, name)                                                      \
  __global__ void name(type *inbuff, type *inountbuff, int n)                            \
  {                                                                                      \
    int idx = GLOBAL_IDX;                                     \
    if (idx < n)                                                                         \
      inountbuff[idx] = (inbuff[idx] < inountbuff[idx]) ? inbuff[idx] : inountbuff[idx]; \
  }

#define MAKE_KERNEL_SUM(type, name)                           \
  __global__ void name(type *inbuff, type *inountbuff, int n) \
  {                                                           \
    int idx = GLOBAL_IDX;          \
    if (idx < n)                                              \
      inountbuff[idx] = inbuff[idx] + inountbuff[idx];        \
  }

#define MAKE_KERNEL_PROD(type, name)                          \
  __global__ void name(type *inbuff, type *inountbuff, int n) \
  {                                                           \
    int idx = GLOBAL_IDX;          \
    if (idx < n)                                              \
      inountbuff[idx] = inbuff[idx] * inountbuff[idx];        \
  }

#define MAKE_KERNEL_LAND(type, name)                          \
  __global__ void name(type *inbuff, type *inountbuff, int n) \
  {                                                           \
    int idx = GLOBAL_IDX;          \
    if (idx < n)                                              \
      inountbuff[idx] = inbuff[idx] && inountbuff[idx];       \
  }

#define MAKE_KERNEL_LOR(type, name)                           \
  __global__ void name(type *inbuff, type *inountbuff, int n) \
  {                                                           \
    int idx = GLOBAL_IDX;          \
    if (idx < n)                                              \
      inountbuff[idx] = inbuff[idx] || inountbuff[idx];       \
  }

#define MAKE_KERNEL_LXOR(type, name)                          \
  __global__ void name(type *inbuff, type *inountbuff, int n) \
  {                                                           \
    int idx = GLOBAL_IDX;          \
    if (idx < n)                                              \
      inountbuff[idx] = inbuff[idx] != inountbuff[idx];       \
  }

#define MAKE_KERNEL_BAND(type, name)                          \
  __global__ void name(type *inbuff, type *inountbuff, int n) \
  {                                                           \
    int idx = GLOBAL_IDX;          \
    if (idx < n)                                              \
      inountbuff[idx] = inbuff[idx] & inountbuff[idx];        \
  }
#define MAKE_KERNEL_BOR(type, name)                           \
  __global__ void name(type *inbuff, type *inountbuff, int n) \
  {                                                           \
    int idx = GLOBAL_IDX;          \
    if (idx < n)                                              \
      inountbuff[idx] = inbuff[idx] | inountbuff[idx];        \
  }
#define MAKE_KERNEL_BXOR(type, name)                          \
  __global__ void name(type *inbuff, type *inountbuff, int n) \
  {                                                           \
    int idx = GLOBAL_IDX;          \
    if (idx < n)                                              \
      inountbuff[idx] = inbuff[idx] ^ inountbuff[idx];        \
  }

// int8
MAKE_KERNEL_SUM(int8_t, sum_int8)
MAKE_KERNEL_PROD(int8_t, prod_int8)
MAKE_KERNEL_MAX(int8_t, max_int8)
MAKE_KERNEL_MIN(int8_t, min_int8)
MAKE_KERNEL_LAND(int8_t, land_int8)
MAKE_KERNEL_BAND(int8_t, band_int8)
MAKE_KERNEL_LOR(int8_t, lor_int8)
MAKE_KERNEL_BOR(int8_t, bor_int8)
MAKE_KERNEL_LXOR(int8_t, lxor_int8)
MAKE_KERNEL_BXOR(int8_t, bxor_int8)

// int16
MAKE_KERNEL_SUM(int16_t, sum_int16)
MAKE_KERNEL_PROD(int16_t, prod_int16)
MAKE_KERNEL_MAX(int16_t, max_int16)
MAKE_KERNEL_MIN(int16_t, min_int16)
MAKE_KERNEL_LAND(int16_t, land_int16)
MAKE_KERNEL_BAND(int16_t, band_int16)
MAKE_KERNEL_LOR(int16_t, lor_int16)
MAKE_KERNEL_BOR(int16_t, bor_int16)
MAKE_KERNEL_LXOR(int16_t, lxor_int16)
MAKE_KERNEL_BXOR(int16_t, bxor_int16)

// int32
MAKE_KERNEL_SUM(int32_t, sum_int32)
MAKE_KERNEL_PROD(int32_t, prod_int32)
MAKE_KERNEL_MAX(int32_t, max_int32)
MAKE_KERNEL_MIN(int32_t, min_int32)
MAKE_KERNEL_LAND(int32_t, land_int32)
MAKE_KERNEL_BAND(int32_t, band_int32)
MAKE_KERNEL_LOR(int32_t, lor_int32)
MAKE_KERNEL_BOR(int32_t, bor_int32)
MAKE_KERNEL_LXOR(int32_t, lxor_int32)
MAKE_KERNEL_BXOR(int32_t, bxor_int32)

// int64
MAKE_KERNEL_SUM(int64_t, sum_int64)
MAKE_KERNEL_PROD(int64_t, prod_int64)
MAKE_KERNEL_MAX(int64_t, max_int64)
MAKE_KERNEL_MIN(int64_t, min_int64)
MAKE_KERNEL_LAND(int64_t, land_int64)
MAKE_KERNEL_BAND(int64_t, band_int64)
MAKE_KERNEL_LOR(int64_t, lor_int64)
MAKE_KERNEL_BOR(int64_t, bor_int64)
MAKE_KERNEL_LXOR(int64_t, lxor_int64)
MAKE_KERNEL_BXOR(int64_t, bxor_int64)

// int
MAKE_KERNEL_SUM(int, sum_int)
MAKE_KERNEL_PROD(int, prod_int)
MAKE_KERNEL_MAX(int, max_int)
MAKE_KERNEL_MIN(int, min_int)
MAKE_KERNEL_LAND(int, land_int)
MAKE_KERNEL_BAND(int, band_int)
MAKE_KERNEL_LOR(int, lor_int)
MAKE_KERNEL_BOR(int, bor_int)
MAKE_KERNEL_LXOR(int, lxor_int)
MAKE_KERNEL_BXOR(int, bxor_int)

// float
MAKE_KERNEL_SUM(float, sum_float)
MAKE_KERNEL_PROD(float, prod_float)
MAKE_KERNEL_MAX(float, max_float)
MAKE_KERNEL_MIN(float, min_float)
MAKE_KERNEL_LAND(float, land_float)
MAKE_KERNEL_LOR(float, lor_float)
MAKE_KERNEL_LXOR(float, lxor_float)

// double
MAKE_KERNEL_SUM(double, sum_double)
MAKE_KERNEL_PROD(double, prod_double)
MAKE_KERNEL_MAX(double, max_double)
MAKE_KERNEL_MIN(double, min_double)
MAKE_KERNEL_LAND(double, land_double)
MAKE_KERNEL_LOR(double, lor_double)
MAKE_KERNEL_LXOR(double, lxor_double)

// char
MAKE_KERNEL_SUM(char, sum_char)
MAKE_KERNEL_PROD(char, prod_char)
MAKE_KERNEL_MAX(char, max_char)
MAKE_KERNEL_MIN(char, min_char)
MAKE_KERNEL_LAND(char, land_char)
MAKE_KERNEL_BAND(char, band_char)
MAKE_KERNEL_LOR(char, lor_char)
MAKE_KERNEL_BOR(char, bor_char)
MAKE_KERNEL_LXOR(char, lxor_char)
MAKE_KERNEL_BXOR(char, bxor_char)

typedef void (*kernel_func)(void *, void *, int);

static inline enum ReduceOp mpi_to_reduce_op(MPI_Op op)
{
  if (MPI_MAX == op)
    return R_MAX;
  if (MPI_MIN == op)
    return R_MIN;
  if (MPI_SUM == op)
    return R_SUM;
  if (MPI_PROD == op)
    return R_PROD;
  if (MPI_LAND == op)
    return R_LAND;
  if (MPI_BAND == op)
    return R_BAND;
  if (MPI_LOR == op)
    return R_LOR;
  if (MPI_BOR == op)
    return R_BOR;
  if (MPI_LXOR == op)
    return R_LXOR;
  if (MPI_BXOR == op)
    return R_BXOR;
  return R_UNNOWN_OP;
}

static inline enum ReduceType mpi_to_redcue_type(MPI_Datatype dtype)
{
  if (MPI_INT8_T == dtype)
    return R_INT8;
  if (MPI_INT16_T == dtype)
    return R_INT16;
  if (MPI_INT32_T == dtype)
    return R_INT32;
  if (MPI_INT64_T == dtype)
    return R_INT64;
  if (MPI_INT == dtype)
    return R_INT;
  if (MPI_FLOAT == dtype)
    return R_FLOAT;
  if (MPI_DOUBLE == dtype)
    return R_DOUBLE;
  if (MPI_CHAR == dtype)
    return R_CHAR;
  return R_UNNOWN_TYPE;
}

kernel_func kernels[R_TYPE_NUM][R_OP_NUM] = {
    {(kernel_func)sum_int8, (kernel_func)prod_int8, (kernel_func)max_int8, (kernel_func)min_int8, (kernel_func)land_int8, (kernel_func)band_int8,
     (kernel_func)lor_int8, (kernel_func)bor_int8, (kernel_func)lxor_int8, (kernel_func)bxor_int8, NULL}, // int8
    {(kernel_func)sum_int16, (kernel_func)prod_int16, (kernel_func)max_int16, (kernel_func)min_int16, (kernel_func)land_int16, (kernel_func)band_int16,
     (kernel_func)lor_int16, (kernel_func)bor_int16, (kernel_func)lxor_int16, (kernel_func)bxor_int16, NULL}, // int16
    {(kernel_func)sum_int32, (kernel_func)prod_int32, (kernel_func)max_int32, (kernel_func)min_int32, (kernel_func)land_int32, (kernel_func)band_int32,
     (kernel_func)lor_int32, (kernel_func)bor_int32, (kernel_func)lxor_int32, (kernel_func)bxor_int32, NULL}, // int32
    {(kernel_func)sum_int64, (kernel_func)prod_int64, (kernel_func)max_int64, (kernel_func)min_int64, (kernel_func)land_int64, (kernel_func)band_int64,
     (kernel_func)lor_int64, (kernel_func)bor_int64, (kernel_func)lxor_int64, (kernel_func)bxor_int64, NULL}, // int64
    {(kernel_func)sum_int, (kernel_func)prod_int, (kernel_func)max_int, (kernel_func)min_int, (kernel_func)land_int, (kernel_func)band_int, (kernel_func)lor_int,
     (kernel_func)bor_int, (kernel_func)lxor_int, (kernel_func)bxor_int, NULL}, // int
    {(kernel_func)sum_float, (kernel_func)prod_float, (kernel_func)max_float, (kernel_func)min_float, (kernel_func)land_float, (kernel_func)NULL,
     (kernel_func)lor_float, NULL, (kernel_func)lxor_float, NULL, NULL}, // float
    {(kernel_func)sum_double, (kernel_func)prod_double, (kernel_func)max_double, (kernel_func)min_double, (kernel_func)land_double, (kernel_func)NULL,
     (kernel_func)lor_double, NULL, (kernel_func)lxor_double, NULL, NULL}, // double
    {(kernel_func)sum_char, (kernel_func)prod_char, (kernel_func)max_char, (kernel_func)min_char, (kernel_func)land_char, (kernel_func)band_char,
     (kernel_func)lor_char, (kernel_func)bor_char, (kernel_func)lxor_char, (kernel_func)bxor_char, NULL}, // char
    {NULL}                                                                                                // unnown type
};

int reduce_wrapper(void *inbuff, void *inoutbuff, int count, MPI_Datatype dtype, MPI_Op op)
{
  enum ReduceOp r_op = mpi_to_reduce_op(op);
  enum ReduceType r_type = mpi_to_redcue_type(dtype);

  if (r_op == R_UNNOWN_OP || r_type == R_UNNOWN_TYPE || r_op >= R_OP_NUM || r_type >= R_TYPE_NUM)
  {
    return MPI_ERR_UNKNOWN;
  }

  int blockSize = count < MAX_THERAD ? count : MAX_THERAD;
  int gridSize = (count + blockSize - 1) / blockSize;

  kernel_func kfunc = kernels[r_type][r_op];
  if (kfunc == NULL)
    return MPI_ERR_UNSUPPORTED_OPERATION;

  kfunc<<<gridSize, blockSize>>>(inbuff, inoutbuff, count);
  cudaError_t err = cudaGetLastError();
  if( err != cudaSuccess ) {
    fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudaDeviceSynchronize();
  return MPI_SUCCESS;
}
