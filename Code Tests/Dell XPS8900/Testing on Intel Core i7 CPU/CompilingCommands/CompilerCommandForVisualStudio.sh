mpiicc -O3 -xCORE-AVX2 -qopenmp -qopt-report-phase=vec -qopt-report=3 -g -debug all -trace fox_floats_timer_caching_omp_fileIO_benchmark.c -o fox_floats_timer_caching_omp_fileIO_benchmark

export VT_PCTRACE=on
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope

mpirun -n -4 -trace ./fox_floats_timer_caching_omp

mpirun -n 4 -trace ./fox_floats_timer_caching_omp_fileIO_benchmark

mpiicc -O3 -xCORE-AVX2 -qopenmp -qopt-report-phase=vec -qopt-report=3 -g -debug all -trace fox_floats_timer_caching_omp_fileIO_benchmark.c -o fox_floats_timer_caching_omp_fileIO_benchmark

mpirun -n 4 -trace ./fox_floats_timer_caching_omp_fileIO_benchmark

mpirun -n 4 -trace ./fox_floats_timer_caching_omp_fileIO_benchmark > mpirun.2048.out

icc -O3 -xCORE-AVX2 -qopt-report-phase=vec -qopt-report=3 serial_matrix_multiplication_benchmark.c -o serial_matrix_multiplication_benchmark

./serial_matrix_multiplication_benchmark > serial.2048.out