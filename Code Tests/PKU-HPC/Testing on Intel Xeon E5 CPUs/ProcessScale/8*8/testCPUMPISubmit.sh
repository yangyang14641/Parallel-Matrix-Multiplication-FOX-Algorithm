#!/bin/bash
#SBATCH -o fox_floats_timer_caching_omp_fileIO_benchmark.%j.%N.out         
#SBATCH -A hpc0006174261     
#SBATCH --partition=C032M0128G                
#SBATCH --qos=normal                         
#SBATCH -J whoami_cpus                        
#SBATCH --get-user-env                        
#SBATCH --nodes=2                         
#SBATCH --ntasks-per-node=32                  
#SBATCH --cpus-per-task=1                    
#SBATCH --mail-type=end                    
#SBATCH --mail-user=yangyang14641@pku.edu.cn    
#SBATCH --time=04:00:00


export OMP_NUM_THREADS=1

srun hostname -s | sort -n > slurm.hosts

module load intel/2018.1

mpiicc -qopenmp -O3 -xCORE-AVX2 -qopt-report-phase=vec -qopt-report=3 -o fox_floats_timer_caching_omp_fileIO_benchmark.o fox_floats_timer_caching_omp_fileIO_benchmark.c

mpirun -n 64 -machinefile slurm.hosts ./fox_floats_timer_caching_omp_fileIO_benchmark.o

rm -rf slurm.hosts
