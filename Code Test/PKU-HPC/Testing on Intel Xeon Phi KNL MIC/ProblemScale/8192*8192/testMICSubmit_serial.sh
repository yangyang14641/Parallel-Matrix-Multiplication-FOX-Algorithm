#!/bin/bash                                           
#SBATCH -o fox_floats_timer_caching_omp_fileIO_benchmark.%j.%N.out       
#SBATCH -A hpc0006174261                  
#SBATCH --partition=KNL                 
#SBATCH --qos=normal               
#SBATCH -J whoami_mic   
#SBATCH --get-user-env  
#SBATCH --nodes=1           
#SBATCH --ntasks-per-node=1                   
#SBATCH --cpus-per-task=1                 
#SBATCH --mail-type=end                                   
#SBATCH --mail-user=yangyang14641@pku.edu.cn                 
#SBATCH --time=04:00:00  

module load intel/2018.1

icc -O3 -xMIC-AVX512 -qopt-report-phase=vec -qopt-report=3 -o serial_matrix_multiplication_benchmark.o serial_matrix_multiplication_benchmark.c

./serial_matrix_multiplication_benchmark.o
