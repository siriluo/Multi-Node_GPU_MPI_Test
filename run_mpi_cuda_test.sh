#!/bin/bash
#SBATCH --job-name=mpi_cuda_test
#SBATCH --account=bbka-dtai-gh
#SBATCH --reservation=update
#SBATCH --partition=ghx4               # DeltaAI GH200 partition (4 GPUs/node)
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4            # 1 MPI rank per GPU
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ── Modules ──────────────────────────────────────────────────────────────────
#module purge
module load cray-mpich/9.0.1


# ── CUDA-aware MPI (GPU Transport Layer) ─────────────────────────────────────
# Required for Cray MPICH to pass GPU device pointers directly through MPI
# without the application staging through host memory.
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_PRELOAD=${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_cuda.so

# ── Optional diagnostics ──────────────────────────────────────────────────────
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export CUDA_VISIBLE_DEVICES=0,1,2,3  # explicit GPU ordering per node

# ── Build (skip if already compiled) ─────────────────────────────────────────
BINARY=./mpi_cuda_test
if [ ! -f "${BINARY}" ]; then
    echo "[build] Compiling mpi_cuda_test.cu ..."
    nvcc -O2 -o mpi_cuda_test mpi_cuda_test.cu \
        -I${MPICH_DIR}/include \
        -L${MPICH_DIR}/lib \
        -lmpi \
        -L${CRAY_MPICH_ROOTDIR}/gtl/lib \
        -lmpi_gtl_cuda
    if [ $? -ne 0 ]; then
        echo "[build] FAILED. Aborting." >&2
        exit 1
    fi
    echo "[build] Done."
fi

# ── Run info ──────────────────────────────────────────────────────────────────
echo "=============================="
echo "Job:     $SLURM_JOB_ID"
echo "Nodes:   $SLURM_NODELIST"
echo "Ranks:   $SLURM_NTASKS  (${SLURM_NTASKS_PER_NODE} per node)"
echo "GPUs:    $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) per node"
echo "Started: $(date)"
echo "=============================="

# ── Execute ───────────────────────────────────────────────────────────────────
srun ${BINARY}

echo ""
echo "Finished: $(date)"
