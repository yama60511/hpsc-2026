#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/barrier>

using namespace std;
namespace cde = cuda::device::experimental;
using barrier = cuda::barrier<cuda::thread_scope_block>;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(status));                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,          \
              static_cast<int>(status));                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CU(call)                                                         \
  do {                                                                         \
    CUresult status = (call);                                                  \
    if (status != CUDA_SUCCESS) {                                              \
      const char *name = nullptr;                                              \
      cuGetErrorName(status, &name);                                           \
      fprintf(stderr, "CU error %s:%d: %s\n", __FILE__, __LINE__,              \
              name ? name : "unknown");                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static constexpr const char *kKernelName = "fp16_256x128_wgmma_tma_s4";
static constexpr int kBlockM = 256;
static constexpr int kBlockN = 128;
static constexpr int kBlockK = 64;
static constexpr int kStages = 4;
static constexpr int kThreads = 512;
static constexpr int kSubTile = 128 * 64;
static constexpr int kATile = 2 * kSubTile;
static constexpr int kBTile = kSubTile;
static constexpr int kStageHalfs = kATile + kBTile;
static constexpr int kWgAStride = 64 * 64;
static constexpr int kLoadBytes = int(sizeof(half) * kStageHalfs);
static constexpr int kSmemBytes = int(sizeof(half) * kStages * kStageHalfs);

__global__ void convert_to_half_kernel(const float *__restrict__ src,
                                       half *__restrict__ dst,
                                       int64_t count) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) dst[idx] = __float2half_rn(src[idx]);
}

__device__ __forceinline__ int swizzle_128b_index(int half_index) {
  int b = half_index * int(sizeof(half));
  b ^= (b & 0x380) >> 3;
  return b / int(sizeof(half));
}

__global__ void prepack_a_tiles(int dim_m, int dim_k,
                                const float *__restrict__ d_a,
                                half *__restrict__ packed_a) {
  int tile_m = blockIdx.x;
  int tile_k = blockIdx.y;
  int tid = threadIdx.x;
  int off_m = tile_m * 128;
  int kk = tile_k * 64;
  half *tile = packed_a + size_t(tile_m * gridDim.y + tile_k) * kSubTile;
  for (int idx = tid; idx < 128 * 64; idx += blockDim.x) {
    int row = idx / 64;
    int k_local = idx % 64;
    int k_in = k_local % 8;
    int k_core = k_local / 8;
    int row_in = row % 8;
    int row_group = row / 8;
    int sidx =
        swizzle_128b_index(row_group * 512 + row_in * 64 + k_core * 8 + k_in);
    int grow = off_m + row;
    tile[sidx] = (grow < dim_m && kk + k_local < dim_k)
                     ? __float2half(d_a[size_t(kk + k_local) * dim_m + grow])
                     : __float2half(0.0f);
  }
}

__global__ void prepack_b_tiles(int dim_n, int dim_k,
                                const float *__restrict__ d_b,
                                half *__restrict__ packed_b) {
  int tile_n = blockIdx.x;
  int tile_k = blockIdx.y;
  int tid = threadIdx.x;
  int off_n = tile_n * 128;
  int kk = tile_k * 64;
  half *tile = packed_b + size_t(tile_n * gridDim.y + tile_k) * kSubTile;
  for (int idx = tid; idx < 64 * 128; idx += blockDim.x) {
    int k_local = idx / 128;
    int col = idx % 128;
    int k_in = k_local % 8;
    int k_core = k_local / 8;
    int col_in = col % 8;
    int col_group = col / 8;
    int sidx =
        swizzle_128b_index(col_group * 512 + col_in * 64 + k_core * 8 + k_in);
    int gcol = off_n + col;
    tile[sidx] = (gcol < dim_n && kk + k_local < dim_k)
                     ? __float2half(d_b[size_t(gcol) * dim_k + kk + k_local])
                     : __float2half(0.0f);
  }
}

static void encode_tile_map(CUtensorMap *map, half *base, uint64_t tile_count) {
  cuuint64_t gdim[2] = {64, tile_count * 128ULL};
  cuuint64_t gstride[1] = {64 * sizeof(half)};
  cuuint32_t bdim[2] = {64, 128};
  cuuint32_t estride[2] = {1, 1};
  CHECK_CU(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, base,
                                  gdim, gstride, bdim, estride,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_NONE,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

__device__ __forceinline__ uint64_t make_wgmma_desc(half *smem, int sbo,
                                                     int lbo, int swizzle) {
  uint64_t desc = 0;
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem)) >> 4;
  desc |= uint64_t(swizzle) << 62;
  desc |= uint64_t(sbo) << 32;
  desc |= uint64_t(lbo) << 16;
  desc |= uint64_t(addr);
  return desc;
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n128k16(uint64_t da, uint64_t db,
                                                  float (&acc)[64]) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
      "%64, %65, 1, 1, 1, 0, 0;\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]), "+f"(acc[4]),
        "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7]), "+f"(acc[8]), "+f"(acc[9]),
        "+f"(acc[10]), "+f"(acc[11]), "+f"(acc[12]), "+f"(acc[13]),
        "+f"(acc[14]), "+f"(acc[15]), "+f"(acc[16]), "+f"(acc[17]),
        "+f"(acc[18]), "+f"(acc[19]), "+f"(acc[20]), "+f"(acc[21]),
        "+f"(acc[22]), "+f"(acc[23]), "+f"(acc[24]), "+f"(acc[25]),
        "+f"(acc[26]), "+f"(acc[27]), "+f"(acc[28]), "+f"(acc[29]),
        "+f"(acc[30]), "+f"(acc[31]), "+f"(acc[32]), "+f"(acc[33]),
        "+f"(acc[34]), "+f"(acc[35]), "+f"(acc[36]), "+f"(acc[37]),
        "+f"(acc[38]), "+f"(acc[39]), "+f"(acc[40]), "+f"(acc[41]),
        "+f"(acc[42]), "+f"(acc[43]), "+f"(acc[44]), "+f"(acc[45]),
        "+f"(acc[46]), "+f"(acc[47]), "+f"(acc[48]), "+f"(acc[49]),
        "+f"(acc[50]), "+f"(acc[51]), "+f"(acc[52]), "+f"(acc[53]),
        "+f"(acc[54]), "+f"(acc[55]), "+f"(acc[56]), "+f"(acc[57]),
        "+f"(acc[58]), "+f"(acc[59]), "+f"(acc[60]), "+f"(acc[61]),
        "+f"(acc[62]), "+f"(acc[63])
      : "l"(da), "l"(db)
      : "memory");
}

__global__ void __launch_bounds__(kThreads, 1)
tensorcore_kernel(int dim_m, int dim_n, int dim_k, int ktile_count,
                  const __grid_constant__ CUtensorMap map_a,
                  const __grid_constant__ CUtensorMap map_b,
                  float *__restrict__ d_c) {
  extern __shared__ __align__(1024) half smem[];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar[kStages];

  int tid = threadIdx.x;
  int wg = tid >> 7;
  int cm = blockIdx.x;
  int cn = blockIdx.y;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) init(&bar[s], kThreads);
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  auto issue = [&](int s, int kt) -> barrier::arrival_token {
    half *base = smem + s * kStageHalfs;
    if (tid == 0) {
      barrier::arrival_token tok =
          cuda::device::barrier_arrive_tx(bar[s], 1, kLoadBytes);
      int a0 = (2 * cm + 0) * ktile_count + kt;
      int a1 = (2 * cm + 1) * ktile_count + kt;
      int b0 = cn * ktile_count + kt;
      cde::cp_async_bulk_tensor_2d_global_to_shared(base, &map_a, 0, a0 * 128,
                                                    bar[s]);
      cde::cp_async_bulk_tensor_2d_global_to_shared(base + kSubTile, &map_a, 0,
                                                    a1 * 128, bar[s]);
      cde::cp_async_bulk_tensor_2d_global_to_shared(base + kATile, &map_b, 0,
                                                    b0 * 128, bar[s]);
      return tok;
    }
    return bar[s].arrive();
  };

  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = 0.0f;

  barrier::arrival_token tok[kStages];
  int prime = (kStages - 1 < ktile_count) ? kStages - 1 : ktile_count;
  for (int s = 0; s < prime; ++s) tok[s] = issue(s, s);

  for (int kt = 0; kt < ktile_count; ++kt) {
    int s = kt % kStages;
    bar[s].wait(std::move(tok[s]));

    int nxt = kt + (kStages - 1);
    if (nxt < ktile_count) tok[nxt % kStages] = issue(nxt % kStages, nxt);

    half *a = smem + s * kStageHalfs;
    half *b = a + kATile;
    wgmma_fence();
    uint64_t da = make_wgmma_desc(a + wg * kWgAStride, 64, 1, 1);
    uint64_t db = make_wgmma_desc(b, 64, 1, 1);
#pragma unroll
    for (int k16 = 0; k16 < kBlockK / 16; ++k16)
      wgmma_m64n128k16(da + 2 * k16, db + 2 * k16, acc);
    wgmma_commit();
    wgmma_wait();
    __syncthreads();
  }

  int lane = tid & 31;
  int warp = (tid & 127) >> 5;
  int row0 = warp * 16 + (lane >> 2);
  int row1 = row0 + 8;
  int col_pair = (lane & 3) * 2;
  int base_row = cm * kBlockM + wg * 64;
  int base_col = cn * kBlockN;
#pragma unroll
  for (int ngrp = 0; ngrp < 16; ++ngrp) {
    int col = base_col + ngrp * 8 + col_pair;
    int r0 = base_row + row0;
    int r1 = base_row + row1;
    d_c[size_t(col + 0) * dim_m + r0] = acc[ngrp * 4 + 0];
    d_c[size_t(col + 1) * dim_m + r0] = acc[ngrp * 4 + 1];
    d_c[size_t(col + 0) * dim_m + r1] = acc[ngrp * 4 + 2];
    d_c[size_t(col + 1) * dim_m + r1] = acc[ngrp * 4 + 3];
  }
}

static bool is_integer_arg(const char *s) {
  if (!s || !*s) return false;
  if (*s == '+' || *s == '-') ++s;
  if (!*s) return false;
  while (*s) {
    if (*s < '0' || *s > '9') return false;
    ++s;
  }
  return true;
}

static void parse_args(int argc, const char **argv, int *m, int *n, int *k,
                       int *nt) {
  *m = 10240;
  *n = 8192;
  *k = 4096;
  *nt = 10;

  if (argc > 1 && strcmp(argv[1], "-h") == 0) {
    printf("usage: %s [m] [k] [n] [Nt]\n", argv[0]);
    printf("       %s %s [m] [n] [k] [Nt]\n", argv[0], kKernelName);
    exit(0);
  }

  if (argc > 1 && is_integer_arg(argv[1])) {
    *m = atoi(argv[1]);
    if (argc > 2) *k = atoi(argv[2]);
    if (argc > 3) *n = atoi(argv[3]);
    if (argc > 4) *nt = atoi(argv[4]);
    return;
  }

  if (argc > 1) {
    if (strcmp(argv[1], kKernelName) != 0) {
      fprintf(stderr, "unknown kernel: %s\n", argv[1]);
      exit(1);
    }
    if (argc > 2) *m = atoi(argv[2]);
    if (argc > 3) *n = atoi(argv[3]);
    if (argc > 4) *k = atoi(argv[4]);
    if (argc > 5) *nt = atoi(argv[5]);
  }
}

static void check_dimensions(int m, int n, int k) {
  if (m % kBlockM || n % kBlockN || k % kBlockK) {
    fprintf(stderr, "dimensions must be divisible by m%%%d, n%%%d, k%%%d\n",
            kBlockM, kBlockN, kBlockK);
    exit(1);
  }
}

static void convert_inputs_to_half(float *A, float *B, half *Ah, half *Bh,
                                   int m, int n, int k) {
  constexpr int threads = 256;
  int64_t asize = int64_t(m) * k;
  int64_t bsize = int64_t(k) * n;
  convert_to_half_kernel<<<(asize + threads - 1) / threads, threads>>>(A, Ah,
                                                                        asize);
  convert_to_half_kernel<<<(bsize + threads - 1) / threads, threads>>>(B, Bh,
                                                                        bsize);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

static double benchmark_cublas(cublasHandle_t handle, int m, int n, int k,
                               int nt, const half *Ah, const half *Bh,
                               float *C) {
  float alpha = 1.0f;
  float beta = 0.0f;
  auto tic = chrono::steady_clock::now();

  for (int i = 0; i < nt + 2; ++i) {
    if (i == 2) tic = chrono::steady_clock::now();
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                              Ah, CUDA_R_16F, m, Bh, CUDA_R_16F, k, &beta, C,
                              CUDA_R_32F, m, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  auto toc = chrono::steady_clock::now();
  return chrono::duration<double>(toc - tic).count() / nt;
}

static double benchmark_tensorcore(int m, int n, int k, int nt, const float *A,
                                   const float *B, half *pA, half *pB,
                                   float *C) {
  int ktile = k / kBlockK;
  int amt = m / 128;
  int bnt = n / 128;
  prepack_a_tiles<<<dim3(amt, ktile), 256>>>(m, k, A, pA);
  prepack_b_tiles<<<dim3(bnt, ktile), 256>>>(n, k, B, pB);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CUtensorMap map_a, map_b;
  encode_tile_map(&map_a, pA, uint64_t(amt) * ktile);
  encode_tile_map(&map_b, pB, uint64_t(bnt) * ktile);

  CHECK_CUDA(cudaFuncSetAttribute(tensorcore_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kSmemBytes));
  dim3 grid(m / kBlockM, n / kBlockN);
  dim3 block(kThreads);
  auto tic = chrono::steady_clock::now();

  for (int i = 0; i < nt + 2; ++i) {
    if (i == 2) tic = chrono::steady_clock::now();
    tensorcore_kernel<<<grid, block, kSmemBytes>>>(m, n, k, ktile, map_a, map_b,
                                                   C);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  auto toc = chrono::steady_clock::now();
  return chrono::duration<double>(toc - tic).count() / nt;
}

int main(int argc, const char **argv) {
  int m, n, k, nt;
  parse_args(argc, argv, &m, &n, &k, &nt);
  check_dimensions(m, n, k);

  printf("kernel: %s, m=%d, n=%d, k=%d, Nt=%d\n", kKernelName, m, n, k, nt);

  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;
  float *C2 = nullptr;
  half *Ah = nullptr;
  half *Bh = nullptr;
  half *pA = nullptr;
  half *pB = nullptr;

  CHECK_CUDA(cudaMallocManaged((void **)&A, int64_t(m) * k * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged((void **)&B, int64_t(k) * n * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged((void **)&C, int64_t(m) * n * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged((void **)&C2, int64_t(m) * n * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&Ah, int64_t(m) * k * sizeof(half)));
  CHECK_CUDA(cudaMalloc((void **)&Bh, int64_t(k) * n * sizeof(half)));
  CHECK_CUDA(cudaMalloc((void **)&pA, int64_t(m / 128) * (k / kBlockK) *
                                          kSubTile * sizeof(half)));
  CHECK_CUDA(cudaMalloc((void **)&pB, int64_t(n / 128) * (k / kBlockK) *
                                          kSubTile * sizeof(half)));

  srand48(0);
  for (int64_t i = 0; i < int64_t(m) * k; ++i) A[i] = drand48();
  for (int64_t i = 0; i < int64_t(k) * n; ++i) B[i] = drand48();
  CHECK_CUDA(cudaMemset(C, 0, int64_t(m) * n * sizeof(float)));
  CHECK_CUDA(cudaMemset(C2, 0, int64_t(m) * n * sizeof(float)));
  convert_inputs_to_half(A, B, Ah, Bh, m, n, k);

  cublasHandle_t cublas_handle;
  CHECK_CUBLAS(cublasCreate(&cublas_handle));
  CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));

  int64_t num_flops =
      2 * int64_t(m) * int64_t(n) * int64_t(k) + 2 * int64_t(m) * n;

  double tcublas = benchmark_cublas(cublas_handle, m, n, k, nt, Ah, Bh, C);
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;

  double tkernel = benchmark_tensorcore(m, n, k, nt, A, B, pA, pB, C2);
  double kernel_flops = double(num_flops) / tkernel / 1.0e9;

  printf("CUBLAS_GEMMEX: %.2f Gflops, WMMA_%s: %.2f Gflops, ratio: %.2f%%\n",
         cublas_flops, kKernelName, kernel_flops,
         100.0 * kernel_flops / cublas_flops);
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops,
         kernel_flops);

  long double sum_abs = 0.0L;
  long double sum_ref_abs = 0.0L;
  double max_abs = 0.0;
  int64_t csize = int64_t(m) * n;
  for (int64_t i = 0; i < csize; ++i) {
    double diff = fabs(double(C[i]) - double(C2[i]));
    sum_abs += diff;
    sum_ref_abs += fabs(double(C[i]));
    if (diff > max_abs) max_abs = diff;
  }
  double avg_abs = double(sum_abs / csize);
  double rel_l1 = double(sum_abs / sum_ref_abs);
  printf("error: %lf\n", avg_abs);
  printf("error_abs_avg: %.6e, error_rel_l1: %.6e, error_abs_max: %.6e\n",
         avg_abs, rel_l1, max_abs);

  CHECK_CUBLAS(cublasDestroy(cublas_handle));
  CHECK_CUDA(cudaFree(A));
  CHECK_CUDA(cudaFree(B));
  CHECK_CUDA(cudaFree(C));
  CHECK_CUDA(cudaFree(C2));
  CHECK_CUDA(cudaFree(Ah));
  CHECK_CUDA(cudaFree(Bh));
  CHECK_CUDA(cudaFree(pA));
  CHECK_CUDA(cudaFree(pB));
  return 0;
}
