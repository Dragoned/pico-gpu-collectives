/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

#ifndef PICO_CORE_UTILS_H
#define PICO_CORE_UTILS_H

#include <mpi.h>
#include <stdio.h>
#if defined PICO_MPI_CUDA_AWARE || defined PICO_NCCL
#include <cuda_runtime.h>
#endif
#ifdef PICO_NCCL
#include <nccl.h>
#endif
#ifdef PICO_UCC
#include <ucc/api/ucc.h>
#include <limits.h>
#include <stdlib.h>
#endif

#include "pico_mpi_nccl_ucc_mapper.h"

#ifdef PICO_UCC
// UCC environment: library, context and team handles
typedef struct
{
  ucc_lib_h lib;
  ucc_context_h ctx;
  ucc_team_h team;
} pico_ucc_env_t;
#endif

#include "libpico.h"

#define PICO_ROOT_RANK 0

#if defined(__GNUC__) || defined(__clang__)
  #define PICO_CORE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
  #define PICO_CORE_UNLIKELY(x) (x)
#endif // defined(__GNUC__) || defined(__clang__)

// Used to print algorithm and collective when in debug mode
#ifndef DEBUG
  #define PICO_CORE_DEBUG_PRINT_STR(name)
  #define PICO_CORE_DEBUG_PRINT_BUFFERS(result, expected, count, dtype, comm, use_barrier) do {} while(0)
#elif !defined PICO_NCCL //&& !defined PICO_MPI_CUDA_AWARE  // Trying this with gpu buffer would need memcpy back to host
  #define PICO_CORE_DEBUG_PRINT_STR(name)           \
    do{                                         \
      int my_r;                                 \
      MPI_Comm_rank(MPI_COMM_WORLD, &my_r);     \
      if(my_r == 0){ printf("%s\n", name); }    \
    } while(0)

  #define PICO_CORE_DEBUG_PRINT_BUFFERS(result, expected, count, dtype, comm, use_barrier)      \
    do {                                                                                    \
    print_buffers(NULL, (result), (expected), 0, (count), (dtype), (comm), (use_barrier));  \
    } while(0)
#endif // DEBUG

#define CHECK_STR(var, name, ret)              \
  if(strcmp(var, name) == 0) {                 \
    PICO_CORE_DEBUG_PRINT_STR(name);               \
    return ret;                                \
  }

#define PICO_CORE_MAX_PATH_LENGTH 512
#define PICO_CORE_BASE_EPSILON_FLOAT 1e-6    // Base epsilon for float
#define PICO_CORE_BASE_EPSILON_DOUBLE 1e-15  // Base epsilon for double

#define PICO_CORE_MAX_ALLOC_NAME_LEN ( MPI_MAX_PROCESSOR_NAME + 64 )
#define PICO_CORE_HEADER_LUMI "MPI_Rank,allocation,xname\n"
#define PICO_CORE_HEADER_DEFAULT "MPI_Rank,allocation\n"

extern size_t bine_allreduce_segsize;

//-----------------------------------------------------------------------------------------------
//                        ENUM FOR TEST DESCRIPTION
// ----------------------------------------------------------------------------------------------

/**
 * @enum coll_t
 *
 * @brief Defines the collective operation to be used in this test. It only provide symbolic
 * name for collective selection.
 * */
typedef enum{
  ALLREDUCE = 0,
  ALLGATHER,
  ALLTOALL,
  ALLTOALLV,
  BCAST,
  GATHER,
  REDUCE,
  REDUCE_SCATTER,
  SCATTER,
  COLL_UNKNOWN
}coll_t;

/**
 * @enum output_level_t
 *
 * @brief Defines the output level of data saving for pico_coremark's results.
 * */
typedef enum{
  ALL = 0,
  STATISTICS,
  MINIMAL,
  SUMMARIZED
}output_level_t;

//-----------------------------------------------------------------------------------------------
//                         ALLOCATOR FUNCTIONS
//-----------------------------------------------------------------------------------------------

#define ALLOCATOR_ARGS    void **sbuf, void **rbuf, void **rbuf_gt, size_t count,\
                          size_t type_size, MPI_Comm comm
/**
* @typedef allocator_func_ptr
*
* A function pointer type for custom memory allocation functions.
*/
typedef int (*allocator_func_ptr)(ALLOCATOR_ARGS);

int allreduce_allocator(ALLOCATOR_ARGS);
int allgather_allocator(ALLOCATOR_ARGS);
int alltoall_allocator(ALLOCATOR_ARGS);
int alltoallv_allocator(ALLOCATOR_ARGS);
int bcast_allocator(ALLOCATOR_ARGS);
int gather_allocator(ALLOCATOR_ARGS);
int reduce_allocator(ALLOCATOR_ARGS);
int reduce_scatter_allocator(ALLOCATOR_ARGS);
int scatter_allocator(ALLOCATOR_ARGS);

//-----------------------------------------------------------------------------------------------
//                               FUNCTION POINTER AND WRAPPER
//                       (for specific collective function and gt_check)
//-----------------------------------------------------------------------------------------------
#if !defined PICO_NCCL && !defined PICO_UCC
typedef int (*allreduce_func_ptr)(ALLREDUCE_MPI_ARGS);
typedef int (*allgather_func_ptr)(ALLGATHER_MPI_ARGS);
typedef int (*alltoall_func_ptr)(ALLTOALL_MPI_ARGS);
typedef int (*alltoallv_func_ptr)(ALLTOALLV_MPI_ARGS);
typedef int (*bcast_func_ptr)(BCAST_MPI_ARGS);
typedef int (*gather_func_ptr)(GATHER_MPI_ARGS);
typedef int (*reduce_func_ptr)(REDUCE_MPI_ARGS);
typedef int (*reduce_scatter_func_ptr)(REDUCE_SCATTER_MPI_ARGS);
typedef int (*scatter_func_ptr)(SCATTER_MPI_ARGS);

static inline int allreduce_wrapper(ALLREDUCE_MPI_ARGS){
  return MPI_Allreduce(sbuf, rbuf, (int)count, dtype, op, comm);
}
static inline int allgather_wrapper(ALLGATHER_MPI_ARGS){
  return MPI_Allgather(sbuf, (int)scount, sdtype, rbuf, (int)rcount, rdtype, comm);
}
static inline int alltoall_wrapper(ALLTOALL_MPI_ARGS){
  return MPI_Alltoall(sbuf, (int)scount, sdtype, rbuf, (int)rcount, rdtype, comm);
}
static inline int alltoallv_wrapper(ALLTOALLV_MPI_ARGS){
  return MPI_Alltoallv(sbuf, scounts, sdispls, sdtype,rbuf, rcounts, rdispls, rdtype, comm);
}
static inline int bcast_wrapper(BCAST_MPI_ARGS){
  return MPI_Bcast(buf, (int)count, dtype, root, comm);
}
static inline int gather_wrapper(GATHER_MPI_ARGS){
  return MPI_Gather(sbuf, (int)scount, sdtype, rbuf, (int)rcount, rdtype, root, comm);
}
static inline int reduce_wrapper(REDUCE_MPI_ARGS){
  return MPI_Reduce(sbuf, rbuf, (int)count, dtype, op, root, comm);
}
static inline int scatter_wrapper(SCATTER_MPI_ARGS){
  return MPI_Scatter(sbuf, (int)scount, sdtype, rbuf, (int)rcount, rdtype, root, comm);
}
#elif defined PICO_UCC
typedef ucc_coll_args_t (*allreduce_func_ptr)(ALLREDUCE_UCC_ARGS);
typedef ucc_coll_args_t (*allgather_func_ptr)(ALLGATHER_UCC_ARGS);
typedef ucc_coll_args_t (*alltoall_func_ptr)(ALLTOALL_UCC_ARGS);
typedef ucc_coll_args_t (*alltoallv_func_ptr)(ALLTOALLV_UCC_ARGS);
typedef ucc_coll_args_t (*bcast_func_ptr)(BCAST_UCC_ARGS);
typedef ucc_coll_args_t (*gather_func_ptr)(GATHER_UCC_ARGS);
typedef ucc_coll_args_t (*reduce_func_ptr)(REDUCE_UCC_ARGS);
typedef ucc_coll_args_t (*reduce_scatter_func_ptr)(REDUCE_SCATTER_UCC_ARGS);
typedef ucc_coll_args_t (*scatter_func_ptr)(SCATTER_UCC_ARGS);

// zero-initialize a ucc_coll_args_t and set its collective type
static inline ucc_coll_args_t pico_ucc_conf_collective_init(ucc_coll_type_t coll_type)
{
  ucc_coll_args_t coll_args;

  memset(&coll_args, 0, sizeof(coll_args));
  coll_args.coll_type = coll_type;

  return coll_args;
}

// configure ucc buffer information
static inline void pico_ucc_conf_buffer(ucc_coll_buffer_info_t *info, const void *buff, size_t count, ucc_datatype_t dtype, ucc_memory_type_t mem_type)
{
  info->buffer = (void *)buff;
  info->count = count;
  info->datatype = dtype;
  info->mem_type = mem_type;
}

// configure ucc vbuffer information
static inline void pico_ucc_conf_bufferv(ucc_coll_buffer_info_v_t *info_v, const void *buff, const ucc_count_t *counts, const ucc_aint_t *displs, ucc_datatype_t dtype, ucc_memory_type_t mem_type)
{
  info_v->buffer = (void *)buff;
  info_v->counts = (ucc_count_t *)counts;
  info_v->displacements = (ucc_aint_t *)displs;
  info_v->datatype = dtype;
  info_v->mem_type = mem_type;
}

// memory type reported to UCC: CUDA buffers with CUDA-aware MPI, host buffers otherwise
#ifdef PICO_MPI_CUDA_AWARE
#define PICO_UCC_MEM_TYPE UCC_MEMORY_TYPE_CUDA
#else
#define PICO_UCC_MEM_TYPE UCC_MEMORY_TYPE_HOST
#endif

static inline ucc_coll_args_t allreduce_wrapper_ucc(ALLREDUCE_UCC_ARGS)
{
  ucc_coll_args_t coll_conf = pico_ucc_conf_collective_init(UCC_COLL_TYPE_ALLREDUCE);

  pico_ucc_conf_buffer(&coll_conf.src.info, sbuf, count, dtype, PICO_UCC_MEM_TYPE);

  pico_ucc_conf_buffer(&coll_conf.dst.info, rbuf, count, dtype, PICO_UCC_MEM_TYPE);
  coll_conf.op = op;
  return coll_conf;
}

static inline ucc_coll_args_t allgather_wrapper_ucc(ALLGATHER_UCC_ARGS)
{
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);
  ucc_coll_args_t coll_conf = pico_ucc_conf_collective_init(UCC_COLL_TYPE_ALLGATHER);

  pico_ucc_conf_buffer(&coll_conf.src.info, sbuf, scount, sdtype, PICO_UCC_MEM_TYPE);

  pico_ucc_conf_buffer(&coll_conf.dst.info, rbuf, rcount, rdtype, PICO_UCC_MEM_TYPE);

  return coll_conf;
}

static inline ucc_coll_args_t alltoall_wrapper_ucc(ALLTOALL_UCC_ARGS)
{
  ucc_coll_args_t coll_conf = pico_ucc_conf_collective_init(UCC_COLL_TYPE_ALLTOALL);

  pico_ucc_conf_buffer(&coll_conf.src.info, sbuf, scount, sdtype, PICO_UCC_MEM_TYPE);

  pico_ucc_conf_buffer(&coll_conf.dst.info, rbuf, rcount, rdtype, PICO_UCC_MEM_TYPE);

  return coll_conf;
}

static inline ucc_coll_args_t alltoallv_wrapper_ucc(ALLTOALLV_UCC_ARGS)
{
  ucc_coll_args_t coll_conf = pico_ucc_conf_collective_init(UCC_COLL_TYPE_ALLTOALLV);
  coll_conf.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll_conf.flags = UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;

  pico_ucc_conf_bufferv(&coll_conf.src.info_v, sbuf, scounts, sdispls, sdtype, PICO_UCC_MEM_TYPE);

  pico_ucc_conf_bufferv(&coll_conf.dst.info_v, rbuf, rcounts, rdispls, rdtype, PICO_UCC_MEM_TYPE);
  return coll_conf;
}

static inline ucc_coll_args_t bcast_wrapper_ucc(BCAST_UCC_ARGS)
{
  ucc_coll_args_t coll_conf = pico_ucc_conf_collective_init(UCC_COLL_TYPE_BCAST);

  pico_ucc_conf_buffer(&coll_conf.src.info, buf, count, dtype, PICO_UCC_MEM_TYPE);

  pico_ucc_conf_buffer(&coll_conf.dst.info, buf, count, dtype, PICO_UCC_MEM_TYPE);

  coll_conf.root = root;
  return coll_conf;
}

static inline ucc_coll_args_t gather_wrapper_ucc(GATHER_UCC_ARGS)
{
  ucc_coll_args_t coll_conf = pico_ucc_conf_collective_init(UCC_COLL_TYPE_GATHER);

  pico_ucc_conf_buffer(&coll_conf.src.info, sbuf, scount, sdtype, PICO_UCC_MEM_TYPE);

  pico_ucc_conf_buffer(&coll_conf.dst.info, rbuf, rcount, rdtype, PICO_UCC_MEM_TYPE);

  coll_conf.root = root;
  return coll_conf;
}

static inline ucc_coll_args_t reduce_scatter_wrapper_ucc(REDUCE_SCATTER_UCC_ARGS)
{
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);
  ucc_count_t total_count = 0;
  for (int i = 0; i < comm_sz; i++)
  {
    total_count += (ucc_count_t)rcounts[i];
  }
  ucc_coll_args_t coll_conf = pico_ucc_conf_collective_init(UCC_COLL_TYPE_REDUCE_SCATTERV);
  coll_conf.mask  = UCC_COLL_ARGS_FIELD_FLAGS;
  coll_conf.flags = UCC_COLL_ARGS_FLAG_COUNT_64BIT;

  pico_ucc_conf_buffer(&coll_conf.src.info, sbuf, total_count, dtype, PICO_UCC_MEM_TYPE);
  pico_ucc_conf_bufferv(&coll_conf.dst.info_v, rbuf, (ucc_count_t *)rcounts, NULL, dtype, PICO_UCC_MEM_TYPE);
  coll_conf.op = op;
  return coll_conf;
}

static inline ucc_coll_args_t reduce_wrapper_ucc(REDUCE_UCC_ARGS)
{
  ucc_coll_args_t coll_conf = pico_ucc_conf_collective_init(UCC_COLL_TYPE_REDUCE);

  pico_ucc_conf_buffer(&coll_conf.src.info, sbuf, count, dtype, PICO_UCC_MEM_TYPE);

  pico_ucc_conf_buffer(&coll_conf.dst.info, rbuf, count, dtype, PICO_UCC_MEM_TYPE);

  coll_conf.op = op;
  coll_conf.root = root;
  return coll_conf;
}

static inline ucc_coll_args_t scatter_wrapper_ucc(SCATTER_UCC_ARGS){
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);

  static ucc_count_t *scnts  = NULL;
  static ucc_aint_t  *sdisps = NULL;
  static int scap = 0;

  if (scap < comm_sz) {
    free(scnts);
    free(sdisps);
    scnts  = (ucc_count_t *)malloc((size_t)comm_sz * sizeof(ucc_count_t));
    sdisps = (ucc_aint_t *)malloc((size_t)comm_sz * sizeof(ucc_aint_t));
    scap = comm_sz;
  }
  for (int i = 0; i < comm_sz; i++) {
    scnts[i]  = (ucc_count_t)scount;
    sdisps[i] = (ucc_aint_t)i * (ucc_aint_t)scount;
  }

  ucc_coll_args_t coll_conf = pico_ucc_conf_collective_init(UCC_COLL_TYPE_SCATTERV);
  coll_conf.mask  = UCC_COLL_ARGS_FIELD_FLAGS;
  coll_conf.flags = UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;

  pico_ucc_conf_bufferv(&coll_conf.src.info_v, sbuf, scnts, sdisps, sdtype, PICO_UCC_MEM_TYPE);
  pico_ucc_conf_buffer(&coll_conf.dst.info, rbuf, rcount, rdtype, PICO_UCC_MEM_TYPE);
  coll_conf.root = root;
  return coll_conf;
}
#else
typedef ncclResult_t (*allreduce_func_ptr)(ALLREDUCE_NCCL_ARGS);
typedef ncclResult_t (*allgather_func_ptr)(ALLGATHER_NCCL_ARGS);
typedef ncclResult_t (*alltoall_func_ptr)(ALLTOALL_NCCL_ARGS);
typedef int (*alltoallv_func_ptr)(ALLTOALLV_NCCL_ARGS);
typedef ncclResult_t (*bcast_func_ptr)(BCAST_NCCL_ARGS);
typedef ncclResult_t (*gather_func_ptr)(GATHER_NCCL_ARGS);
typedef ncclResult_t (*reduce_func_ptr)(REDUCE_NCCL_ARGS);
typedef ncclResult_t (*reduce_scatter_func_ptr)(REDUCE_SCATTER_NCCL_ARGS);
typedef ncclResult_t (*scatter_func_ptr)(SCATTER_NCCL_ARGS);
#endif


//-----------------------------------------------------------------------------------------------
//                                TEST ROUTINE STRUCTURE
//-----------------------------------------------------------------------------------------------

/**
 * @struct test_routine_t
 * @brief Structure to hold collective type and function pointers
 * for collective specific allocator, custom collective and 
 * ground truth functions pointers.
 * It also holds the output level, the output folder path and the data folder path.
 *
 * @var collective Specifies the type of collective operation.
 * @var allocator Pointer to the memory allocator function.
 * @var function Union of function pointers for allreduce, allgather and reduce scatter.
 */
typedef struct {
  coll_t collective; /**< Specifies the type of collective operation. */
  allocator_func_ptr allocator; /**< Pointer to the memory allocator function. */
#if defined PICO_MPI_CUDA_AWARE || defined PICO_NCCL
  allocator_func_ptr allocator_cuda; /**< Pointer to the CUDA memory allocator function. */
#endif // PICO_MPI_CUDA_AWARE || PICO_NCCL
  size_t segsize; /**< Size of the segment for segmented collectives. */

  /** Union of function pointers for custom collective functions. */
  union {
    allreduce_func_ptr allreduce;
    allgather_func_ptr allgather;
    alltoall_func_ptr alltoall;
    alltoallv_func_ptr alltoallv;
    bcast_func_ptr bcast;
    gather_func_ptr gather;
    reduce_func_ptr reduce;
    reduce_scatter_func_ptr reduce_scatter;
    scatter_func_ptr scatter;
  } function;

  output_level_t output_level;    /**< Specifies the output level for data saving. */
  char output_data_file[PICO_CORE_MAX_PATH_LENGTH];   /**< Path to the output directory. */
  char alloc_file[PICO_CORE_MAX_PATH_LENGTH];         /**< Path to the data directory. */
} test_routine_t;


// ----------------------------------------------------------------------------------------------
//                                CUDA FUNCTIONS
// ----------------------------------------------------------------------------------------------

#if defined PICO_MPI_CUDA_AWARE || defined PICO_NCCL

int allreduce_allocator_cuda(ALLOCATOR_ARGS);
int allgather_allocator_cuda(ALLOCATOR_ARGS);
int alltoall_allocator_cuda(ALLOCATOR_ARGS);
int alltoallv_allocator_cuda(ALLOCATOR_ARGS);
int bcast_allocator_cuda(ALLOCATOR_ARGS);
int gather_allocator_cuda(ALLOCATOR_ARGS);
int reduce_allocator_cuda(ALLOCATOR_ARGS);
int reduce_scatter_allocator_cuda(ALLOCATOR_ARGS);
int scatter_allocator_cuda(ALLOCATOR_ARGS);

int coll_memcpy_host_to_device(void** d_buf, void** buf, size_t count, size_t type_size, coll_t coll);
int coll_memcpy_device_to_host(void** d_buf, void** buf, size_t count, size_t type_size, coll_t coll);

#endif

//-----------------------------------------------------------------------------------------------
//                          UCC FUNCTIONS
//-----------------------------------------------------------------------------------------------

#ifdef PICO_UCC
// out-of-band (OOB) bootstrap request: wraps the MPI request used by UCC for peer discovery
typedef struct
{
  MPI_Request mpi_req;
} pico_ucc_oob_req_t;

// OOB allgather: nonblocking MPI_Iallgather used by UCC to exchange bootstrap data;
// coll_info carries the MPI_Comm, req is the heap-allocated request wrapper
static inline ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                         void *coll_info, void **req)
{
  MPI_Comm comm = (MPI_Comm)(uintptr_t)coll_info;
  pico_ucc_oob_req_t *oob_req;

  if (PICO_CORE_UNLIKELY(msglen > INT_MAX))
  {
    return UCC_ERR_INVALID_PARAM;
  }

  oob_req = malloc(sizeof(*oob_req));
  if (PICO_CORE_UNLIKELY(!oob_req))
  {
    return UCC_ERR_NO_MEMORY;
  }

  MPI_Iallgather(sbuf, (int)msglen, MPI_BYTE, rbuf, (int)msglen, MPI_BYTE,
                 comm, &oob_req->mpi_req);
  *req = oob_req;
  return UCC_OK;
}

// poll the OOB request: UCC_OK when the underlying MPI_Iallgather completed
static inline ucc_status_t oob_allgather_test(void *req)
{
  MPI_Request *mpi_req = &((pico_ucc_oob_req_t *)req)->mpi_req;
  int completed;

  MPI_Test(mpi_req, &completed, MPI_STATUS_IGNORE);
  return completed ? UCC_OK : UCC_INPROGRESS;
}

// release the OOB request wrapper
static inline ucc_status_t oob_allgather_free(void *req)
{
  free(req);
  return UCC_OK;
}

// initialize the UCC environment: library, then context (with OOB over the MPI communicator),
// then team; on failure unwinds via goto and returns -1
static inline int pico_ucc_env_init(pico_ucc_env_t *env, MPI_Comm comm, int size, int rank)
{
  ucc_lib_config_h lib_config = NULL;
  ucc_context_config_h ctx_config = NULL;
  ucc_lib_params_t lib_params;
  ucc_context_params_t ctx_params;
  ucc_team_params_t team_params;
  ucc_status_t status;

  /* 1. Initialize UCC library */
  memset(&lib_params, 0, sizeof(lib_params));
  lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode = UCC_THREAD_SINGLE;

  if (UCC_OK != ucc_lib_config_read(NULL, NULL, &lib_config))
  {
    fprintf(stderr, "Error: pico_ucc_env_init: ucc_lib_config_read failed\n");
    goto err_lib_config;
  }

  if (UCC_OK != ucc_init(&lib_params, lib_config, &env->lib))
  {
    fprintf(stderr, "Error: pico_ucc_env_init: ucc_init failed\n");
    goto err_init;
  }

  ucc_lib_config_release(lib_config);
  lib_config = NULL;

  /* 2. Create UCC context */
  memset(&ctx_params, 0, sizeof(ctx_params));
  ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;
  ctx_params.oob.allgather = oob_allgather;
  ctx_params.oob.req_test = oob_allgather_test;
  ctx_params.oob.req_free = oob_allgather_free;
  ctx_params.oob.coll_info = (void *)(uintptr_t)comm;
  ctx_params.oob.n_oob_eps = size;
  ctx_params.oob.oob_ep = rank;

  if (UCC_OK != ucc_context_config_read(env->lib, NULL, &ctx_config))
  {
    fprintf(stderr, "Error: pico_ucc_env_init: ucc_context_config_read failed\n");
    goto err_lib;
  }

  if (UCC_OK != ucc_context_create(env->lib, &ctx_params, ctx_config, &env->ctx))
  {
    fprintf(stderr, "Error: pico_ucc_env_init: ucc_context_create failed\n");
    goto err_ctx_config;
  }

  ucc_context_config_release(ctx_config);
  ctx_config = NULL;

  /* 3. Create UCC team */
  memset(&team_params, 0, sizeof(team_params));
  team_params.mask = UCC_TEAM_PARAM_FIELD_EP |
                     UCC_TEAM_PARAM_FIELD_EP_RANGE |
                     UCC_TEAM_PARAM_FIELD_OOB;
  team_params.ep = rank;
  team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
  team_params.oob.allgather = oob_allgather;
  team_params.oob.req_test = oob_allgather_test;
  team_params.oob.req_free = oob_allgather_free;
  team_params.oob.coll_info = (void *)(uintptr_t)comm;
  team_params.oob.n_oob_eps = size;
  team_params.oob.oob_ep = rank;

  if (UCC_OK != ucc_team_create_post(&env->ctx, 1, &team_params, &env->team))
  {
    fprintf(stderr, "Error: pico_ucc_env_init: ucc_team_create_post failed\n");
    goto err_ctx;
  }

  while (UCC_INPROGRESS == (status = ucc_team_create_test(env->team)))
  {
    ucc_context_progress(env->ctx);
  }

  if (status != UCC_OK)
  {
    fprintf(stderr, "Error: pico_ucc_env_init: ucc_team_create_test failed (status=%d)\n", (int)status);
    goto err_ctx;
  }

  return 0;

err_ctx:
  ucc_context_destroy(env->ctx);
err_ctx_config:
  if (NULL != ctx_config) ucc_context_config_release(ctx_config);
err_lib:
  ucc_finalize(env->lib);
err_init:
  if (NULL != lib_config) ucc_lib_config_release(lib_config);
err_lib_config:
  return -1;
}

// tear down the UCC environment in reverse order: team, context, library
static inline void pico_ucc_env_finalize(pico_ucc_env_t *env)
{
  ucc_team_destroy(env->team);
  ucc_context_destroy(env->ctx);
  ucc_finalize(env->lib);
}

#endif

//-----------------------------------------------------------------------------------------------
//                                TAG INITIALIZATION HELPER
//-----------------------------------------------------------------------------------------------
#if defined PICO_INSTRUMENT && defined PICO_UCC
#error "PICO_INSTRUMENT is not supported with PICO_UCC"
#endif

#if defined PICO_INSTRUMENT && !defined PICO_NCCL && !defined PICO_UCC

/**
 * @brief Run the selected operation once. Used for tag initialization.
 *
 * @return MPI_SUCCESS on success, an MPI_ERR code on error.
 */
int run_coll_once(test_routine_t test_routine, void *sbuf, void *rbuf,
                   size_t count, MPI_Datatype dtype, MPI_Comm comm);

#define PICO_INSTRUMENTATION_CALLS do {                                 \
  if (libpico_snapshot_store(i) != 0) {                                 \
    fprintf(stderr, "Error: Failed to store snapshot. Aborting...\n");  \
    return -1;                                                          \
  } if (libpico_clear_tags() != 0) {                                    \
    fprintf(stderr, "Error: Failed to clear tags. Aborting...\n");      \
    return -1;                                                          \
  }                                                                     \
} while(0)
#else
#define PICO_INSTRUMENTATION_CALLS
#endif

//-----------------------------------------------------------------------------------------------
//                                MAIN PICO_COREMARK LOOP FUNCTIONS
//-----------------------------------------------------------------------------------------------

#if !defined PICO_NCCL && !defined PICO_UCC
 /**
 * @brief Test loop interface that select the appropriate collective operation
 * test loop based on the collective type and algorithm specified in the test_routine.
 *
 * @return MPI_SUCCESS on success, an MPI_ERR code on error.
 */
int test_loop(test_routine_t test_routine, void *sbuf, void *rbuf, size_t count,
              MPI_Datatype dtype, MPI_Comm comm, int iter, double *times);

/**
 * @macro DEFINE_TEST_LOOP
 * @brief Macro to generate the MPI test loop for a given collective operation.
 *
 * @param OP_NAME Name of the operation.
 * @param ARGS Arguments for the operation.
 * @param COLLECTIVE Collective operation to perform.
 */
#define DEFINE_TEST_LOOP(OP_NAME, ARGS, COLLECTIVE)                   \
static inline int OP_NAME##_test_loop(ARGS, int iter, double *times,  \
                                   test_routine_t test_routine) {         \
  int ret = MPI_SUCCESS;                                                  \
  double start_time, end_time;                                            \
  MPI_Barrier(comm);                                                      \
  for(int i = 0; i < iter; i++) {                                         \
    start_time = MPI_Wtime();                                             \
    ret = test_routine.function.COLLECTIVE;                               \
    end_time = MPI_Wtime();                                               \
    times[i] = end_time - start_time;                                     \
    if(PICO_CORE_UNLIKELY(ret != MPI_SUCCESS)) {                          \
      char error_string[MPI_MAX_ERROR_STRING];                            \
      int result_len;                                                     \
      MPI_Error_string(ret, error_string, &result_len);                   \
      fprintf(stderr, "Error: " #OP_NAME " failed: %s", error_string);    \
      return ret;                                                         \
    }                                                                     \
    PICO_INSTRUMENTATION_CALLS;                                           \
    MPI_Barrier(comm);                                                    \
  }                                                                       \
  return ret;                                                             \
}

DEFINE_TEST_LOOP(allreduce, ALLREDUCE_MPI_ARGS, allreduce(sbuf, rbuf, count, dtype, MPI_SUM, comm))
DEFINE_TEST_LOOP(allgather, ALLGATHER_MPI_ARGS, allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm))
DEFINE_TEST_LOOP(alltoall, ALLTOALL_MPI_ARGS, alltoall(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm))
DEFINE_TEST_LOOP(alltoallv, ALLTOALLV_MPI_ARGS, alltoallv(sbuf, scounts, sdispls, sdtype, rbuf, rcounts, rdispls, rdtype, comm))
DEFINE_TEST_LOOP(bcast, BCAST_MPI_ARGS, bcast(buf, count, dtype, PICO_ROOT_RANK, comm))
DEFINE_TEST_LOOP(gather, GATHER_MPI_ARGS, gather(sbuf, scount, sdtype, rbuf, rcount, rdtype, PICO_ROOT_RANK, comm))
DEFINE_TEST_LOOP(reduce, REDUCE_MPI_ARGS, reduce(sbuf, rbuf, count, dtype, MPI_SUM, PICO_ROOT_RANK, comm))
DEFINE_TEST_LOOP(reduce_scatter, REDUCE_SCATTER_MPI_ARGS, reduce_scatter(sbuf, rbuf, rcounts, dtype, MPI_SUM, comm))
DEFINE_TEST_LOOP(scatter, SCATTER_MPI_ARGS, scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, PICO_ROOT_RANK, comm))
#elif defined PICO_UCC
 /**
 * @brief Test loop interface that select the appropriate collective operation
 * test loop based on the collective type and algorithm specified in the test_routine.
 *
 * @return MPI_SUCCESS on success, an MPI_ERR code on error.
 */
int test_loop(test_routine_t test_routine, void *sbuf, void *rbuf, size_t count,
              ucc_datatype_t dtype, MPI_Comm comm, pico_ucc_env_t ucc_env, int iter, double *times);

#define DEFINE_TEST_LOOP(OP_NAME, ARGS, COLLECTIVE)                                                                           \
  static inline int OP_NAME##_test_loop(ARGS, int iter, double *times,                                                        \
                                        test_routine_t test_routine)                                                          \
  {                                                                                                                           \
    int ret = MPI_SUCCESS;                                                                                                    \
    double start_time, end_time;                                                                                              \
    ucc_coll_req_h coll_req;                                                                                                  \
    ucc_status_t status;                                                                                                      \
    for (int i = 0; i < iter; i++)                                                                                            \
    {                                                                                                                         \
      ucc_coll_args_t coll_conf = test_routine.function.COLLECTIVE;                                                           \
      status = ucc_collective_init(&coll_conf, &coll_req, ucc_env.team);                                                      \
      if (status != UCC_OK)                                                                                                   \
      {                                                                                                                       \
        ret = MPI_ERR_OTHER;                                                                                                  \
        goto OP_NAME##_ucc_fail;                                                                                              \
      }                                                                                                                       \
      MPI_Barrier(comm);                                                                                                      \
      start_time = MPI_Wtime();                                                                                               \
      status = ucc_collective_post(coll_req);                                                                                 \
      if (status != UCC_OK)                                                                                                   \
      {                                                                                                                       \
        ucc_collective_finalize(coll_req);                                                                                    \
        ret = MPI_ERR_OTHER;                                                                                                  \
        goto OP_NAME##_ucc_fail;                                                                                              \
      }                                                                                                                       \
      do                                                                                                                      \
      {                                                                                                                       \
        ucc_context_progress(ucc_env.ctx);                                                                                    \
        status = ucc_collective_test(coll_req);                                                                               \
      } while (status == UCC_INPROGRESS);                                                                                     \
      end_time = MPI_Wtime();                                                                                                 \
      if (status != UCC_OK)                                                                                                   \
      {                                                                                                                       \
        ucc_collective_finalize(coll_req);                                                                                    \
        ret = MPI_ERR_OTHER;                                                                                                  \
        goto OP_NAME##_ucc_fail;                                                                                              \
      }                                                                                                                       \
      status = ucc_collective_finalize(coll_req);                                                                             \
      if (status != UCC_OK)                                                                                                   \
      {                                                                                                                       \
        fprintf(stderr, "Warning: " #OP_NAME " ucc_collective_finalize failed (status=%d)\n", (int)status);                   \
      }                                                                                                                       \
      times[i] = end_time - start_time;                                                                                       \
      if (PICO_CORE_UNLIKELY(ret != MPI_SUCCESS))                                                                             \
      {                                                                                                                       \
        char error_string[MPI_MAX_ERROR_STRING];                                                                              \
        int result_len;                                                                                                       \
        MPI_Error_string(ret, error_string, &result_len);                                                                     \
        fprintf(stderr, "Error: " #OP_NAME " failed: %s", error_string);                                                      \
        return ret;                                                                                                           \
      }                                                                                                                       \
      PICO_INSTRUMENTATION_CALLS;                                                                                             \
      MPI_Barrier(comm);                                                                                                      \
    }                                                                                                                         \
    return ret;                                                                                                               \
    OP_NAME##_ucc_fail:                                                                                                       \
    {                                                                                                                         \
      fprintf(stderr, "Error: " #OP_NAME " UCC collective failed: %s (status=%d)\n", ucc_status_string(status), (int)status); \
      return ret;                                                                                                             \
    }                                                                                                                         \
  }

DEFINE_TEST_LOOP(allreduce, ALLREDUCE_UCC_ARGS, allreduce(sbuf, rbuf, count, dtype, UCC_OP_SUM, comm, ucc_env))
DEFINE_TEST_LOOP(allgather, ALLGATHER_UCC_ARGS, allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm, ucc_env))
DEFINE_TEST_LOOP(alltoall, ALLTOALL_UCC_ARGS, alltoall(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm, ucc_env))
DEFINE_TEST_LOOP(alltoallv, ALLTOALLV_UCC_ARGS, alltoallv(sbuf, scounts, sdispls, sdtype, rbuf, rcounts, rdispls, rdtype, comm, ucc_env))
DEFINE_TEST_LOOP(bcast, BCAST_UCC_ARGS, bcast(buf, count, dtype, PICO_ROOT_RANK, comm, ucc_env))
DEFINE_TEST_LOOP(gather, GATHER_UCC_ARGS, gather(sbuf, scount, sdtype, rbuf, rcount, rdtype, PICO_ROOT_RANK, comm, ucc_env))
DEFINE_TEST_LOOP(reduce, REDUCE_UCC_ARGS, reduce(sbuf, rbuf, count, dtype, UCC_OP_SUM, PICO_ROOT_RANK, comm, ucc_env))
DEFINE_TEST_LOOP(reduce_scatter, REDUCE_SCATTER_UCC_ARGS, reduce_scatter(sbuf, rbuf, rcounts, dtype, UCC_OP_SUM, comm, ucc_env))
DEFINE_TEST_LOOP(scatter, SCATTER_UCC_ARGS, scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, PICO_ROOT_RANK, comm, ucc_env))
#else

 /**
 * @brief Test loop interface that select the appropriate collective operation
 * test loop based on the collective type and algorithm specified in the test_routine.
 *
 * @return 0 on success, -1 on error.
 */
int test_loop(test_routine_t test_routine, void *sbuf, void *rbuf, size_t count,
              ncclDataType_t dtype, ncclComm_t nccl_comm, cudaStream_t stream, int iter, double *times);

/**
 * @macro DEFINE_TEST_LOOP
 * @brief Macro to generate the NCCL test loop for a given collective operation.
 *
 * @param OP_NAME Name of the operation.
 * @param ARGS Arguments for the operation.
 * @param COLLECTIVE Collective operation to perform.
 */
#define DEFINE_TEST_LOOP(OP_NAME, ARGS, COLLECTIVE)                         \
static inline int OP_NAME##_test_loop(ARGS, int iter, double *times,        \
                                      test_routine_t test_routine) {        \
  ncclResult_t ret = ncclSuccess;                                           \
  cudaError_t  cerr;                                                        \
  cudaEvent_t  ev0, ev1;                                                    \
  PICO_CORE_CUDA_CHECK(cudaEventCreate(&ev0), cerr);                        \
  PICO_CORE_CUDA_CHECK(cudaEventCreate(&ev1), cerr);                        \
                                                                            \
  PICO_CORE_CUDA_CHECK(cudaStreamSynchronize(stream), cerr);                \
  MPI_Barrier(MPI_COMM_WORLD);                                              \
  for (int i = 0; i < iter; i++) {                                          \
    PICO_CORE_CUDA_CHECK(cudaEventRecord(ev0, stream), cerr);               \
    ret = test_routine.function.COLLECTIVE;                                 \
    if (PICO_CORE_UNLIKELY(ret != ncclSuccess)) {                           \
      fprintf(stderr, "Error: " #OP_NAME " failed (NCCL): %s\n",            \
              ncclGetErrorString(ret));                                     \
      PICO_CORE_CUDA_CHECK(cudaEventDestroy(ev0), cerr);                    \
      PICO_CORE_CUDA_CHECK(cudaEventDestroy(ev1), cerr);                    \
      return -1;                                                            \
    }                                                                       \
    PICO_CORE_CUDA_CHECK(cudaEventRecord(ev1, stream), cerr);               \
    PICO_CORE_CUDA_CHECK(cudaEventSynchronize(ev1), cerr);                  \
    float ms = 0.0f;                                                        \
    PICO_CORE_CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1), cerr);        \
    times[i] = ms * 1e-3;                                                   \
    MPI_Barrier(MPI_COMM_WORLD);                                            \
  }                                                                         \
  PICO_CORE_CUDA_CHECK(cudaEventDestroy(ev0), cerr);                        \
  PICO_CORE_CUDA_CHECK(cudaEventDestroy(ev1), cerr);                        \
  return 0;                                                                 \
}

DEFINE_TEST_LOOP(allreduce, ALLREDUCE_NCCL_ARGS, allreduce(sbuf, rbuf, count, dtype, ncclSum, nccl_comm, stream))
DEFINE_TEST_LOOP(allgather, ALLGATHER_NCCL_ARGS, allgather(sbuf, rbuf, count, dtype, nccl_comm, stream))
DEFINE_TEST_LOOP(alltoall, ALLTOALL_NCCL_ARGS, alltoall(sbuf, rbuf, count, dtype, nccl_comm, stream))
DEFINE_TEST_LOOP(alltoallv, ALLTOALLV_NCCL_ARGS, alltoallv(sbuf, scounts, sdispls, rbuf, rcounts, rdispls, dtype, nccl_comm, stream))
DEFINE_TEST_LOOP(bcast, BCAST_NCCL_ARGS, bcast(buf, count, dtype, PICO_ROOT_RANK, nccl_comm, stream))
DEFINE_TEST_LOOP(gather, GATHER_NCCL_ARGS, gather(sbuf, rbuf, count, dtype, PICO_ROOT_RANK, nccl_comm, stream))
DEFINE_TEST_LOOP(reduce, REDUCE_NCCL_ARGS, reduce(sbuf, rbuf, count, dtype, ncclSum, PICO_ROOT_RANK, nccl_comm, stream))
DEFINE_TEST_LOOP(reduce_scatter, REDUCE_SCATTER_NCCL_ARGS, reduce_scatter(sbuf, rbuf, rcount, dtype, ncclSum, nccl_comm, stream))
DEFINE_TEST_LOOP(scatter, SCATTER_NCCL_ARGS, scatter(sbuf, rbuf, count, dtype, PICO_ROOT_RANK, nccl_comm, stream))

#endif

//-----------------------------------------------------------------------------------------------
//                                   GROUND TRUTH CHECK FUNCTIONS
//-----------------------------------------------------------------------------------------------

/**
 * @brief Compares two buffers with an epsilon tolerance for float or double datatypes.
 *
 * @param buf_1 First buffer.
 * @param buf_2 Second buffer.
 * @param count Size of the buffers in number of elements.
 * @param dtype PICO_DTYPE_T of the recvbuf.
 * @param comm Communicator.
 * @return 0 if buffers are equal within tolerance, -1 otherwise.
 */
int are_equal_eps(const void *buf_1, const void *buf_2, size_t count,
                  MPI_Datatype dtype, MPI_Comm comm);


/**
 * @macro GT_CHECK_BUFFER
 * @brief Macro to check the result of an MPI operation against the ground truth.
 *
 * It is used inside the ground truth check functions to compare the result of an MPI operation
 * against the ground truth. It checks if the result is equal to the expected value within an
 * epsilon tolerance for float and double datatypes, and uses `memcmp` for other datatypes.
 */
#define GT_CHECK_BUFFER(result, expected, count, dtype, comm)                 \
  do {                                                                        \
    if(dtype != MPI_DOUBLE && dtype != MPI_FLOAT) {                           \
      if(memcmp((result), (expected), (count) * type_size) != 0) {            \
        PICO_CORE_DEBUG_PRINT_BUFFERS((result), (expected), (count), (dtype), (comm), (1));  \
        fprintf(stderr, "Error: results are not valid. Aborting...");         \
        ret = -1;                                                             \
      }                                                                       \
    } else {                                                                  \
      if(are_equal_eps((result), (expected), (count), dtype, comm) == -1) {  \
        PICO_CORE_DEBUG_PRINT_BUFFERS((result), (expected), (count), (dtype), (comm), (1));  \
        fprintf(stderr, "Error: results are not valid. Aborting...");       \
        ret = -1;                                                             \
      }                                                                       \
    }                                                                         \
  } while(0)


/**
 * @brief Interface for ground-truth check functions.
 * This function selects the appropriate ground-truth check function based on the
 * collective type specified in the test_routine.
 *
 * @return 0 on success, an -1 on error.
 */
int ground_truth_check(test_routine_t test_routine, void *sbuf, void *rbuf, void *rbuf_gt,
                       size_t count, MPI_Datatype dtype, MPI_Comm comm);


//-----------------------------------------------------------------------------------------------
//                     SELECT ALGORITHM AND COMMAND LINE PARSING FUNCTIONS
//-----------------------------------------------------------------------------------------------

/**
 * @brief Populates a `test_routine_t` structure based on environment variables
 * and command-line arguments.
 *
 * This function reads the `COLLECTIVE_TYPE` environment variable and reads the 
 * `algorithm` command-line argument to popuplate the `test_routine_t` structure
 * with the appropriate collective type and function pointers.
 *
 * @param test_routine Pointer to a `test_routine_t` structure to populate.
 * @param algorithm The algorithm name as a string.
 *
 * @return `0` on success, `-1` on error.
 */
int get_routine(test_routine_t *test_routine, const char *algorithm);


/**
 * @brief Parses command-line arguments and extracts parameters.
 *
 * @param argc Number of arguments.
 * @param argv Argument vector.
 * @param[out] array_count Size of the array.
 * @param[out] iter Number of iterations.
 * @param[out] algprithm Algorithm name.
 * @param[out] type_string Data type as a string.
 * @return 0 on success, -1 on error.
 */
int get_command_line_arguments(int argc, char** argv, size_t *array_count,
                               int* iter, const char **algorithm, const
                               char **type_string);


/**
 * @brief Retrieves the data saving options based on environment variables.
 *
 * This function reads the `OUTPUT_LEVEL` environment variable to determine
 * the data saving options for the pico_coremark results as well as the
 * `OUTPUT_DIR` and `DATA_DIR` environment variables to set the output
 * directory and data directory respectively.
 *
 * @param test_routine Pointer to a `test_routine_t` structure to populate.
 *
 * @return `0` on success, `-1` on error.
 */
int get_data_saving_options(test_routine_t *test_routine, size_t count,
                            const char *algorithm, const char *type_string);


/**
 * @brief Splits the MPI communicator into inter and intra communicators.
 *
 * This function splits the MPI_COMM_WORLD communicator into two communicators:
 * - `intra_comm`: for communication within a node.
 * - `inter_comm`: for communication between nodes.
 *
 * @param[out] inter_comm Pointer to the inter communicator.
 * @param[out] intra_comm Pointer to the intra communicator.
 *
 * @return MPI_SUCCESS on success, an MPI_ERR code on error.
 *
 * @note The number of tasks per node is determined by the `CURRENT_TASKS_PER_NODE`
 * environment variable. The function checks if the number of ranks is divisible by this value.
 */
int split_communicator(MPI_Comm *inter_comm, MPI_Comm *intra_comm);

//-----------------------------------------------------------------------------------------------
//                                  I/O FUNCTIONS
//-----------------------------------------------------------------------------------------------

#if defined PICO_INSTRUMENT && !defined PICO_NCCL && !defined PICO_UCC

static inline int pico_name_col_width(const char **names, int n, int min_w, int max_w) {
    int w = min_w;
    for (int i = 0; i < n; ++i) {
        int len = (int)strlen(names[i]);
        if (len > w) w = len;
    }
    if (w > max_w) w = max_w;
    return w;
}

/**
 * @brief Writes the instrumentation timing results to a specified output file in CSV format.
 *
 * @param test_routine The test routine structure containing the output file path.
 * @param times An array containing the timing values for each iteration.
 * @param tag_times A 2D array containing the timing values for each tag across all iterations.
 * @param tag_names An array of strings containing the names of the tags.
 * @param iter The number of iterations.
 *
 * @return int Returns 0 on success, or -1 if an error occurs.
 *
 * @note Time is saved in ns (i.e. 10^-9 s).
 */
int write_instrument_output_to_file(test_routine_t test_routine, double* times, double** tag_times, const char** tag_names, int iter);
#endif

/**
 * @brief Writes the timing results to a specified output file in CSV format.
 *
 * The ammount of data to save is determined by the `output_level` parameter.
 * "full" (or its "all" alias) saves every rank for every iteration,
 * "statistics" saves cross-rank statistics for every iteration, "minimal"
 * saves the highest rank time for every iteration, and "summarized" saves one
 * aggregate statistics row after discarding the first 20 percent of samples.
 *
 * @param test_routine The test routine structure containing the output level and file path
 * @param highest An array containing the highest timing values for each iteration.
 * @param all_times A 2D array flattened into 1D containing timing values for all ranks 
 *                  across all iterations.
 * @param iter The number of iterations.
 *
 * @return int Returns 0 on success, or -1 if an error occurs.
 *
 * @note Time is saved in ns (i.e. 10^-9 s).
 */
int write_output_to_file(test_routine_t test_routine, double *highest, double *all_times, int iter);

/**
 * @brief Checks if a file does not exists.
 *
 * @param filename The name of the file to check.
 * @return int Returns 1 if the file does not exists, 0 otherwise.
 */
int file_not_exists(const char* filename);


/**
 * @brief Writes MPI rank and processor name allocations to a specified file using MPI I/O.
 *
 * This function collects the MPI rank and processor name for each process and writes
 * them to a file in CSV format using parallel I/O. The file will have the format:
 *
 * rank,allocation
 * 0,processor_name_0
 * 1,processor_name_1
 * ...
 *
 * Each rank calculates its unique offset using a fixed entry size, based on MPI_MAX_PROCESSOR_NAME,
 * ensuring non-overlapping writes without requiring data size communication or gathering.
 *
 * This implementation uses `MPI_File_write_at` for concurrent, safe file access and a barrier
 * to synchronize ranks after writing the header.
 *
 * @param filename The name of the file to which the allocations will be written.
 * @param comm The MPI communicator.
 *
 * @return int Returns MPI_SUCCESS on success, MPI_ERR otherwise.
 */
int write_allocations_to_file(const char* filename, MPI_Comm comm);


//-----------------------------------------------------------------------------------------------
//                             GENERAL UTILITY FUNCTIONS
//-----------------------------------------------------------------------------------------------

/**
 * @brief Generates a random sbuf based on the specified type, size and collective.
 *
 * @param sbuf Pointer to the sbuf to fill with random values.
 * @param dtype Datatype of the sendbuffer (MPI Datatype).
 * @param array_size Number of elements in the array.
 * @param comm MPI communicator.
 * @param test_routine Routine decision structure.
 *
 * @return 0 on success, -1 if the data type is unsupported.
 */
int rand_sbuf_generator(void *sbuf, MPI_Datatype dtype, size_t array_size,
                         MPI_Comm comm, test_routine_t test_routine);


/**
 * @brief Concatenates a directory path and a filename into a full file path.
 *
 * @param dir_path Directory path.
 * @param filename Filename to append.
 * @param fullpath Buffer where the concatenated path is stored.
 * @return 0 on success, -1 on error.
 */
int concatenate_path(const char *dirpath, const char *filename, char *fullpath);

/**
 * @brief Prints the contents of two buffers for debugging purposes.
 *
 * @param rbuf The buffer to print.
 * @param rbuf_gt The ground-truth buffer to print.
 * @param count The number of elements in the buffer.
 * @param dtype The MPI datatype of the buffer.
 * @param comm The MPI communicator.
 * @param use_barrier Flag to indicate if a barrier should be used.
 */
void print_buffers(const void *sbuf, const void *rbuf, const void *rbuf_gt,
                   size_t sbuf_count, size_t rbuf_count, MPI_Datatype dtype,
                   MPI_Comm comm, int use_barrier);

//-----------------------------------------------------------------------------------------------
//                          DEBUGGING FUNCTIONS
//-----------------------------------------------------------------------------------------------
#ifdef DEBUG
/**
 * @brief Generates the send buffer with a sequence of powers of 10^rank.
 *
 * @param sbuf Pointer to the sbuf to fill with random values.
 * @param dtype Datatype of the sendbuffer (MPI Datatype).
 * @param count Number of elements in the array.
 * @param comm MPI communicator.
 * @param test_routine Routine decision structure.
 *
 * @return 0 on success, -1 if the data type is unsupported.
 */
int debug_sbuf_generator(void *sbuf, MPI_Datatype dtype, size_t count,
                    MPI_Comm comm, test_routine_t test_routine);

#endif // DEBUG

#endif // PICO_CORE_TOOLS_H