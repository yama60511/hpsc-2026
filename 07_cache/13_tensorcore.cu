#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace std;

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

static constexpr const char *kKernelName = "fp16_256x128_ptx_v8l_w2x2_s4";
static constexpr int kBlockM = 256;
static constexpr int kBlockN = 128;
static constexpr int kBlockK = 32;
static constexpr int kStages = 4;
static constexpr int kWarpsM = 2;
static constexpr int kWarpsN = 2;
static constexpr int kThreads = kWarpsM * kWarpsN * 32;
static constexpr int kLda = kBlockM + 8;
static constexpr int kLdb = kBlockK + 8;
static constexpr int kSmemMainBytes =
    int(sizeof(half) * kStages * (kBlockK * kLda + kBlockN * kLdb));
static constexpr int kSmemEpiBytes = int(sizeof(half) * kBlockM * kBlockN);
static constexpr int kSmemBytes =
    kSmemMainBytes > kSmemEpiBytes ? kSmemMainBytes : kSmemEpiBytes;

__global__ void convert_to_half_kernel(const float *__restrict__ src,
                                       half *__restrict__ dst,
                                       int64_t count) {
  int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) dst[idx] = __float2half_rn(src[idx]);
}

__device__ __forceinline__ void cp_async_16(void *dst_shared,
                                            const void *src_global) {
  unsigned int smem_addr =
      static_cast<unsigned int>(__cvta_generic_to_shared(dst_shared));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(src_global)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ half unpack_half2_bits(uint32_t bits, int high) {
  union {
    uint32_t u32;
    __half2 h2;
  } v;
  v.u32 = bits;
  return high ? __high2half(v.h2) : __low2half(v.h2);
}

__device__ __forceinline__ void
mma_m16n8k16_f16(uint32_t (&d)[2], const uint32_t (&a)[4],
                 const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
        "r"(b[1]));
}

__device__ __forceinline__ uint32_t shared_u32_addr(const void *ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr,
                                                  uint32_t (&r)[4]) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t addr,
                                            uint32_t (&r)[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
               "{%0, %1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

__device__ __forceinline__ void
load_mma_a_m16n8k16_ldm_col_smem(const half *base, uint32_t (&a)[4]) {
  int lane = threadIdx.x & 31;
  int col = (lane & 7) + ((lane >> 4) * 8);
  int row_block = ((lane >> 3) & 1) * 8;
  ldmatrix_x4_trans(shared_u32_addr(base + col * kLda + row_block), a);
}

__device__ __forceinline__ void
load_mma_b_m16n8k16_ldm_col_smem(const half *base, uint32_t (&b)[2]) {
  int lane = threadIdx.x & 31;
  int col = lane & 7;
  int row_block = ((lane >> 3) & 1) * 8;
  ldmatrix_x2(shared_u32_addr(base + col * kLdb + row_block), b);
}

__device__ __forceinline__ void
store_mma_m16n8_acc_to_smem(half *epi, int m_base, int n_base,
                            const uint32_t (&d)[2]) {
  int lane = threadIdx.x & 31;
  int group = lane >> 2;
  int tid_in_group = lane & 3;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = (i < 2) ? group : group + 8;
    int col = tid_in_group * 2 + (i & 1);
    epi[(n_base + col) * kBlockM + m_base + row] =
        unpack_half2_bits(d[i >> 1], i & 1);
  }
}

__device__ __forceinline__ void
stage_fp16_cp_async(int tid, int k0, int dim_m, int dim_k, int block_m,
                    int block_n, const half *__restrict__ d_a,
                    const half *__restrict__ d_b, half *tile_a,
                    half *tile_b) {
  constexpr int A_CHUNKS = (kBlockM * kBlockK) / 8;
  constexpr int A_CHUNKS_PER_K = kBlockM / 8;
  for (int idx = tid; idx < A_CHUNKS; idx += kThreads) {
    int kk = idx / A_CHUNKS_PER_K;
    int mm = (idx - kk * A_CHUNKS_PER_K) * 8;
    cp_async_16(tile_a + kk * kLda + mm,
                d_a + (k0 + kk) * dim_m + block_m + mm);
  }

  constexpr int B_CHUNKS = (kBlockN * kBlockK) / 8;
  constexpr int B_CHUNKS_PER_N = kBlockK / 8;
  for (int idx = tid; idx < B_CHUNKS; idx += kThreads) {
    int nn = idx / B_CHUNKS_PER_N;
    int kk = (idx - nn * B_CHUNKS_PER_N) * 8;
    cp_async_16(tile_b + nn * kLdb + kk,
                d_b + (block_n + nn) * dim_k + k0 + kk);
  }
  cp_async_commit();
}

__global__ void __launch_bounds__(kThreads, 1)
tensorcore_kernel(int dim_m, int dim_n, int dim_k,
                  const half *__restrict__ d_a,
                  const half *__restrict__ d_b, float *__restrict__ d_c) {
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;
  constexpr int WARP_M = kBlockM / kWarpsM;
  constexpr int WARP_N = kBlockN / kWarpsN;
  constexpr int ROW_TILES = WARP_M / WMMA_M;
  constexpr int COL_TILES = WARP_N / WMMA_N;
  constexpr int K_TILES = kBlockK / WMMA_K;
  constexpr int GROUP_M = 8;

  extern __shared__ __align__(16) half smem[];
  half *tile_a = smem;
  half *tile_b = tile_a + kStages * kBlockK * kLda;
  half *epi = smem;

  int num_pid_m = gridDim.x;
  int num_pid_n = gridDim.y;
  int pid = blockIdx.x + blockIdx.y * num_pid_m;
  int num_pid_in_group = GROUP_M * num_pid_n;
  int group_id = pid / num_pid_in_group;
  int first_pid_m = group_id * GROUP_M;
  int group_size_m = num_pid_m - first_pid_m;
  if (group_size_m > GROUP_M) group_size_m = GROUP_M;
  int pid_m = first_pid_m + (pid % group_size_m);
  int pid_n = (pid % num_pid_in_group) / group_size_m;

  int block_m = pid_m * kBlockM;
  int block_n = pid_n * kBlockN;
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int warp_m = warp_id / kWarpsN;
  int warp_n = warp_id - warp_m * kWarpsN;

  uint32_t acc[ROW_TILES][COL_TILES][2][2];
#pragma unroll
  for (int row = 0; row < ROW_TILES; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILES; ++col) {
#pragma unroll
      for (int half_n = 0; half_n < 2; ++half_n) {
        acc[row][col][half_n][0] = 0;
        acc[row][col][half_n][1] = 0;
      }
    }
  }

  int num_tiles = dim_k / kBlockK;
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < num_tiles) {
      stage_fp16_cp_async(tid, s * kBlockK, dim_m, dim_k, block_m, block_n,
                          d_a, d_b, tile_a + s * kBlockK * kLda,
                          tile_b + s * kBlockN * kLdb);
    } else {
      cp_async_commit();
    }
  }

  for (int tile = 0; tile < num_tiles; ++tile) {
    cp_async_wait_group<kStages - 2>();
    __syncthreads();

    int next = tile + kStages - 1;
    if (next < num_tiles) {
      int next_stage = next % kStages;
      stage_fp16_cp_async(tid, next * kBlockK, dim_m, dim_k, block_m, block_n,
                          d_a, d_b, tile_a + next_stage * kBlockK * kLda,
                          tile_b + next_stage * kBlockN * kLdb);
    } else {
      cp_async_commit();
    }

    int stage = tile % kStages;
#pragma unroll
    for (int kk = 0; kk < K_TILES; ++kk) {
      uint32_t b_frag[COL_TILES][2][2];
#pragma unroll
      for (int col = 0; col < COL_TILES; ++col) {
#pragma unroll
        for (int half_n = 0; half_n < 2; ++half_n) {
          const half *b_base =
              tile_b + (stage * kBlockN + warp_n * WARP_N + col * WMMA_N +
                        half_n * 8) *
                           kLdb +
              kk * WMMA_K;
          load_mma_b_m16n8k16_ldm_col_smem(b_base, b_frag[col][half_n]);
        }
      }

#pragma unroll
      for (int row = 0; row < ROW_TILES; ++row) {
        uint32_t a_frag[4];
        const half *a_base =
            tile_a + (stage * kBlockK + kk * WMMA_K) * kLda +
            warp_m * WARP_M + row * WMMA_M;
        load_mma_a_m16n8k16_ldm_col_smem(a_base, a_frag);

#pragma unroll
        for (int col = 0; col < COL_TILES; ++col) {
#pragma unroll
          for (int half_n = 0; half_n < 2; ++half_n) {
            mma_m16n8k16_f16(acc[row][col][half_n], a_frag,
                             b_frag[col][half_n]);
          }
        }
      }
    }
  }

  __syncthreads();

#pragma unroll
  for (int row = 0; row < ROW_TILES; ++row) {
#pragma unroll
    for (int col = 0; col < COL_TILES; ++col) {
#pragma unroll
      for (int half_n = 0; half_n < 2; ++half_n) {
        int e_m = warp_m * WARP_M + row * WMMA_M;
        int e_n = warp_n * WARP_N + col * WMMA_N + half_n * 8;
        store_mma_m16n8_acc_to_smem(epi, e_m, e_n, acc[row][col][half_n]);
      }
    }
  }
  __syncthreads();

  for (int idx = tid; idx < kBlockM * kBlockN; idx += kThreads) {
    int cc = idx / kBlockM;
    int rr = idx - cc * kBlockM;
    d_c[(block_n + cc) * dim_m + block_m + rr] = __half2float(epi[idx]);
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
    fprintf(stderr,
            "dimensions must be divisible by m%%%d, n%%%d, k%%%d\n",
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
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                              &alpha, Ah, CUDA_R_16F, m, Bh, CUDA_R_16F, k,
                              &beta, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  auto toc = chrono::steady_clock::now();
  return chrono::duration<double>(toc - tic).count() / nt;
}

static double benchmark_tensorcore(int m, int n, int k, int nt, const half *Ah,
                                   const half *Bh, float *C) {
  CHECK_CUDA(cudaFuncSetAttribute(tensorcore_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kSmemBytes));
  dim3 grid(m / kBlockM, n / kBlockN);
  dim3 block(kThreads);
  auto tic = chrono::steady_clock::now();

  for (int i = 0; i < nt + 2; ++i) {
    if (i == 2) tic = chrono::steady_clock::now();
    tensorcore_kernel<<<grid, block, kSmemBytes>>>(m, n, k, Ah, Bh, C);
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

  CHECK_CUDA(cudaMallocManaged((void **)&A, int64_t(m) * k * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged((void **)&B, int64_t(k) * n * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged((void **)&C, int64_t(m) * n * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged((void **)&C2, int64_t(m) * n * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&Ah, int64_t(m) * k * sizeof(half)));
  CHECK_CUDA(cudaMalloc((void **)&Bh, int64_t(k) * n * sizeof(half)));

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

  double tkernel = benchmark_tensorcore(m, n, k, nt, Ah, Bh, C2);
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
  return 0;
}
