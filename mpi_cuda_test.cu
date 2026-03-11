/*
 * mpi_cuda_test.cu
 *
 * Multi-node GPU MPI communication test for GH200 + Cray MPICH (CUDA-aware).
 *
 * Tests:
 *   1. Topology  – print node name, MPI rank, and assigned GPU per rank
 *   2. Ring      – correctness: ring-shift a GPU buffer one step, verify value
 *   3. Latency   – ping-pong between rank 0 and rank 1, small messages
 *   4. Bandwidth – sweep message sizes from 4 B to 256 MB, report GB/s
 *   5. Allreduce – GPU-buffer collective, verify sum across all ranks
 *
 * Compile:
 *   nvcc -O2 -o mpi_cuda_test mpi_cuda_test.cu \
 *       -I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi \
 *       -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_cuda
 *
 * Environment (must be set before srun):
 *   export MPICH_GPU_SUPPORT_ENABLED=1
 *   export LD_PRELOAD=${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_cuda.so
 */

#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/* ── helpers ─────────────────────────────────────────────────────────────── */

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA] %s:%d  %s\n",                             \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

#define MPI_CHECK(call)                                                        \
    do {                                                                       \
        int _e = (call);                                                       \
        if (_e != MPI_SUCCESS) {                                               \
            char msg[256]; int len;                                            \
            MPI_Error_string(_e, msg, &len);                                   \
            fprintf(stderr, "[MPI] %s:%d  %s\n", __FILE__, __LINE__, msg);    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

/* ── kernels ─────────────────────────────────────────────────────────────── */

__global__ void k_fill(float *buf, float val, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

__global__ void k_count_errors(const float *buf, float expected,
                                int n, int *errs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && fabsf(buf[i] - expected) > 1e-5f)
        atomicAdd(errs, 1);
}

static void fill_gpu(float *d, float val, int n)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    k_fill<<<blocks, threads>>>(d, val, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static int count_errors_gpu(const float *d, float expected, int n)
{
    int *d_errs, h_errs = 0;
    CUDA_CHECK(cudaMalloc(&d_errs, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_errs, 0, sizeof(int)));
    int threads = 256, blocks = (n + threads - 1) / threads;
    k_count_errors<<<blocks, threads>>>(d, expected, n, d_errs);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_errs, d_errs, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_errs));
    return h_errs;
}

/* ── test 1 : topology ───────────────────────────────────────────────────── */

/* Per-rank payload gathered to rank 0 for ordered printing */
typedef struct {
    char   hostname[256];
    char   gpu_name[256];
    int    gpu_idx;
    int    gpu_count;
    size_t gpu_mem_mb;
} TopoInfo;

static void test_topology(int rank, int size)
{
    TopoInfo local;
    gethostname(local.hostname, sizeof(local.hostname));

    CUDA_CHECK(cudaGetDevice(&local.gpu_idx));
    CUDA_CHECK(cudaGetDeviceCount(&local.gpu_count));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, local.gpu_idx));
    strncpy(local.gpu_name, prop.name, sizeof(local.gpu_name) - 1);
    local.gpu_name[sizeof(local.gpu_name) - 1] = '\0';
    local.gpu_mem_mb = prop.totalGlobalMem / (1024 * 1024);

    TopoInfo *all = NULL;
    if (rank == 0)
        all = (TopoInfo *)malloc(size * sizeof(TopoInfo));

    MPI_CHECK(MPI_Gather(&local, sizeof(TopoInfo), MPI_BYTE,
                         all,   sizeof(TopoInfo), MPI_BYTE,
                         0, MPI_COMM_WORLD));

    if (rank == 0) {
        for (int r = 0; r < size; r++)
            printf("  rank %3d | host %-44s | GPU %d/%d  %s  (%zu MB)\n",
                   r, all[r].hostname,
                   all[r].gpu_idx, all[r].gpu_count,
                   all[r].gpu_name, all[r].gpu_mem_mb);
        free(all);
    }
}

/* ── test 2 : ring-shift correctness ─────────────────────────────────────── */

static void test_ring(int rank, int size)
{
    const int N = 1024;  /* floats */
    float *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(float)));

    /* Each rank fills its buffer with its own rank value */
    fill_gpu(d_buf, (float)rank, N);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    MPI_CHECK(MPI_Sendrecv_replace(
        d_buf, N, MPI_FLOAT,
        next, 0,
        prev, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    /* Expected: each rank should have received prev's value */
    float expected = (float)prev;
    int errs = count_errors_gpu(d_buf, expected, N);

    int total_errs;
    MPI_CHECK(MPI_Reduce(&errs, &total_errs, 1, MPI_INT, MPI_SUM, 0,
                         MPI_COMM_WORLD));

    if (rank == 0) {
        if (total_errs == 0)
            printf("  PASSED  (ring-shift, %d elements per rank)\n", N);
        else
            printf("  FAILED  (%d element mismatches)\n", total_errs);
    }

    CUDA_CHECK(cudaFree(d_buf));
}

/* ── test 3 : ping-pong latency ──────────────────────────────────────────── */

#define LAT_ITERS   1000
#define LAT_WARMUP   100

static void test_latency(int rank, int size)
{
    if (size < 2) {
        if (rank == 0) printf("  SKIPPED (need >= 2 ranks)\n");
        return;
    }

    float *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, sizeof(float)));
    fill_gpu(d_buf, 0.0f, 1);

    MPI_Barrier(MPI_COMM_WORLD);

    double t_start = 0.0, t_end = 0.0;

    if (rank == 0) {
        for (int i = 0; i < LAT_WARMUP + LAT_ITERS; i++) {
            if (i == LAT_WARMUP) t_start = MPI_Wtime();
            MPI_CHECK(MPI_Send(d_buf, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD));
            MPI_CHECK(MPI_Recv(d_buf, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE));
        }
        t_end = MPI_Wtime();
        double lat_us = (t_end - t_start) / (2.0 * LAT_ITERS) * 1e6;
        printf("  latency  rank 0 <-> rank 1 : %.2f us  (1 float, %d iters)\n",
               lat_us, LAT_ITERS);
    } else if (rank == 1) {
        for (int i = 0; i < LAT_WARMUP + LAT_ITERS; i++) {
            MPI_CHECK(MPI_Recv(d_buf, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE));
            MPI_CHECK(MPI_Send(d_buf, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(d_buf));
}

/* ── test 4 : bandwidth sweep ────────────────────────────────────────────── */

#define BW_ITERS   50
#define BW_WARMUP  10

static const size_t BW_SIZES[] = {
    4,
    1024,
    64 * 1024,
    1 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024
};
#define N_BW_SIZES (int)(sizeof(BW_SIZES) / sizeof(BW_SIZES[0]))

static void test_bandwidth(int rank, int size)
{
    if (size < 2) {
        if (rank == 0) printf("  SKIPPED (need >= 2 ranks)\n");
        return;
    }

    /* Allocate for the largest message size */
    size_t max_bytes = BW_SIZES[N_BW_SIZES - 1];
    char *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, max_bytes));
    CUDA_CHECK(cudaMemset(d_buf, 0, max_bytes));

    if (rank == 0)
        printf("  %-14s  %10s  %10s\n", "msg size", "bytes", "GB/s");

    for (int s = 0; s < N_BW_SIZES; s++) {
        size_t nbytes = BW_SIZES[s];
        int    nfloat = (int)(nbytes / sizeof(float));
        if (nfloat < 1) nfloat = 1;

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = 0.0, t_end = 0.0;

        if (rank == 0) {
            for (int i = 0; i < BW_WARMUP + BW_ITERS; i++) {
                if (i == BW_WARMUP) t_start = MPI_Wtime();
                MPI_CHECK(MPI_Send(d_buf, nfloat, MPI_FLOAT, 1, 1,
                                   MPI_COMM_WORLD));
                MPI_CHECK(MPI_Recv(d_buf, nfloat, MPI_FLOAT, 1, 1,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            }
            t_end = MPI_Wtime();

            double elapsed   = t_end - t_start;
            /* bidirectional: 2 * nfloat * sizeof(float) bytes per round-trip */
            double bw_gbs    = (2.0 * BW_ITERS * nbytes) / elapsed / 1e9;
            const char *unit = nbytes < 1024 ? "B"
                             : nbytes < 1024 * 1024 ? "KB"
                             : nbytes < 1024 * 1024 * 1024 ? "MB" : "GB";
            double disp = nbytes < 1024 ? (double)nbytes
                        : nbytes < 1024 * 1024 ? nbytes / 1024.0
                        : nbytes / (1024.0 * 1024.0);
            printf("  %8.0f %-5s  %10zu  %10.2f\n",
                   disp, unit, nbytes, bw_gbs);
        } else if (rank == 1) {
            for (int i = 0; i < BW_WARMUP + BW_ITERS; i++) {
                MPI_CHECK(MPI_Recv(d_buf, nfloat, MPI_FLOAT, 0, 1,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                MPI_CHECK(MPI_Send(d_buf, nfloat, MPI_FLOAT, 0, 1,
                                   MPI_COMM_WORLD));
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    CUDA_CHECK(cudaFree(d_buf));
}

/* ── test 5 : GPU allreduce ──────────────────────────────────────────────── */

static void test_allreduce(int rank, int size)
{
    const int N = 1024 * 1024;  /* 4 MB */
    float *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(float)));
    fill_gpu(d_buf, 1.0f, N);   /* each rank contributes 1.0 per element */

    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, d_buf, N, MPI_FLOAT, MPI_SUM,
                            MPI_COMM_WORLD));

    float expected = (float)size;
    int errs = count_errors_gpu(d_buf, expected, N);

    int total_errs;
    MPI_CHECK(MPI_Reduce(&errs, &total_errs, 1, MPI_INT, MPI_SUM, 0,
                         MPI_COMM_WORLD));

    if (rank == 0) {
        if (total_errs == 0)
            printf("  PASSED  (allreduce sum=%.0f, %d floats, %d ranks)\n",
                   expected, N, size);
        else
            printf("  FAILED  (%d element mismatches, expected sum=%.0f)\n",
                   total_errs, expected);
    }

    CUDA_CHECK(cudaFree(d_buf));
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank, size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    /* Assign one GPU per rank (round-robin within the node) */
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % dev_count));

    if (rank == 0) {
        printf("========================================\n");
        printf("  Multi-Node GPU MPI Test\n");
        printf("  Ranks: %d   GPUs/node: %d\n", size, dev_count);
        printf("========================================\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ── 1. Topology ── */
    if (rank == 0) printf("[1] Topology\n");
    test_topology(rank, size);
    if (rank == 0) printf("\n");

    /* ── 2. Ring correctness ── */
    if (rank == 0) printf("[2] Ring-shift correctness\n");
    test_ring(rank, size);
    if (rank == 0) printf("\n");

    /* ── 3. Ping-pong latency ── */
    if (rank == 0) printf("[3] Ping-pong latency (rank 0 <-> rank 1)\n");
    test_latency(rank, size);
    if (rank == 0) printf("\n");

    /* ── 4. Bandwidth sweep ── */
    if (rank == 0) printf("[4] Bandwidth sweep (rank 0 <-> rank 1, bidirectional)\n");
    test_bandwidth(rank, size);
    if (rank == 0) printf("\n");

    /* ── 5. Allreduce ── */
    if (rank == 0) printf("[5] GPU Allreduce\n");
    test_allreduce(rank, size);
    if (rank == 0) printf("\n");

    if (rank == 0) printf("All tests complete.\n");

    MPI_CHECK(MPI_Finalize());
    return 0;
}
