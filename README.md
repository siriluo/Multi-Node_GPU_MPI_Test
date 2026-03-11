# Multi-Node GPU MPI Test

Verifies MPI communication across multiple nodes and GPUs on GH200 using
Cray MPICH with CUDA-aware (GPU-direct) transfers. All MPI calls pass raw
device pointers — no explicit host staging.

---

## Files

| File | Description |
|---|---|
| `mpi_cuda_test.cu` | Test program (5 sub-tests) |
| `run_mpi_cuda_test.sh` | SLURM job script for DeltaAI GH200 |

---

## Test Program: `mpi_cuda_test.cu`

Five sequential tests are run. All MPI communication uses GPU device pointers
directly, exercising Cray MPICH's GPU Transport Layer (GTL).

### Test 1 — Topology
Prints node name, MPI rank, assigned GPU index, GPU model, and memory for
every rank in rank order. Confirms each rank has a unique GPU and that the
node/GPU mapping is correct across nodes.

### Test 2 — Ring-shift Correctness
Each rank fills a 1024-element GPU buffer with its own rank value, then calls
`MPI_Sendrecv_replace` to shift the buffer one step around a logical ring
(rank → rank+1). After the exchange, each rank verifies that all 1024 elements
equal the value from the previous rank. A global reduction collects the total
error count.

### Test 3 — Ping-pong Latency
Rank 0 and rank 1 exchange a single `float` in a ping-pong loop (1000
iterations after 100 warm-up iterations). Reports one-way latency in
microseconds. Measures raw inter-GPU latency, which crosses node boundaries
when ranks 0 and 1 are on different nodes.

### Test 4 — Bandwidth Sweep
Rank 0 and rank 1 exchange buffers of increasing size (4 B → 256 MB) using a
send/recv ping-pong (50 iterations + 10 warm-up). Reports bidirectional
bandwidth in GB/s for each message size.

| Size |
|---|
| 4 B |
| 1 KB |
| 64 KB |
| 1 MB |
| 16 MB |
| 64 MB |
| 256 MB |

### Test 5 — GPU Allreduce
Every rank allocates a 1 M-element GPU buffer initialized to `1.0` and calls
`MPI_Allreduce(MPI_IN_PLACE, ..., MPI_SUM)`. The expected result is `(float)size`
for every element. A kernel-side error check verifies correctness.

### Expected output (8 ranks across 2 nodes)

```
========================================
  Multi-Node GPU MPI Test
  Ranks: 8   GPUs/node: 4
========================================

[1] Topology
  rank   0 | host node001               | GPU 0/4  NVIDIA GH200 96GB  (98304 MB)
  rank   1 | host node001               | GPU 1/4  NVIDIA GH200 96GB  (98304 MB)
  ...
  rank   7 | host node002               | GPU 3/4  NVIDIA GH200 96GB  (98304 MB)

[2] Ring-shift correctness
  PASSED  (ring-shift, 1024 elements per rank)

[3] Ping-pong latency (rank 0 <-> rank 1)
  latency  rank 0 <-> rank 1 :  3.45 us  (1 float, 1000 iters)

[4] Bandwidth sweep (rank 0 <-> rank 1, bidirectional)
  msg size            bytes        GB/s
       4 B               4        0.01
       1 KB            1024        0.12
      64 KB           65536        4.83
       1 MB         1048576       18.45
      16 MB        16777216       24.10
      64 MB        67108864       25.30
     256 MB       268435456       25.60

[5] GPU Allreduce
  PASSED  (allreduce sum=8, 1048576 floats, 8 ranks)

All tests complete.
```

---

## Compile

Load modules:

```bash
module load cray-mpich
module load cuda   # or whichever CUDA module is available on your cluster
```

Compile:

```bash
nvcc -O2 -o mpi_cuda_test mpi_cuda_test.cu \
    -I${MPICH_DIR}/include \
    -L${MPICH_DIR}/lib \
    -lmpi \
    -L${CRAY_MPICH_ROOTDIR}/gtl/lib \
    -lmpi_gtl_cuda
```

`-lmpi_gtl_cuda` links the GPU Transport Layer that enables CUDA-aware paths
inside Cray MPICH. Both env vars (`MPICH_DIR`, `CRAY_MPICH_ROOTDIR`) are set
automatically by `module load cray-mpich`. `nvcc` must be on `PATH` from
whichever CUDA module is loaded.

---

## Run with SLURM

Edit `run_mpi_cuda_test.sh` to set your allocation account, then submit:

```bash
# Set your account
sed -i 's/<your_project>/my_alloc/' run_mpi_cuda_test.sh

sbatch run_mpi_cuda_test.sh
```

Monitor:

```bash
squeue -u $USER
tail -f mpi_cuda_test_<JOBID>.out
```

The script compiles automatically if the binary is missing.

### Key environment variables set by the script

| Variable | Value | Purpose |
|---|---|---|
| `MPICH_GPU_SUPPORT_ENABLED` | `1` | Enable CUDA-aware paths in Cray MPICH |
| `LD_PRELOAD` | `libmpi_gtl_cuda.so` | Inject GTL at runtime |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | Explicit GPU ordering per node |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Segfault / hang in `MPI_Sendrecv_replace` | GTL not loaded | Confirm `MPICH_GPU_SUPPORT_ENABLED=1` and `LD_PRELOAD` is correct |
| `libmpi_gtl_cuda.so: cannot open shared object` | Wrong `CRAY_MPICH_ROOTDIR` | Run `module show cray-mpich` and update path |
| Ring test FAILED | Wrong GPU assigned | Verify `dev_count` matches `--ntasks-per-node` |
| Allreduce FAILED | GTL collective not enabled | Try `MPICH_GPU_SUPPORT_ENABLED=2` (forces GTL for collectives) |
| `mpi.h` not found at compile | Module not loaded | `module load cray-mpich` before `nvcc` |
