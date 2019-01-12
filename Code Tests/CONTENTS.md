# Code Tests Contents

**Code Tests results in Dell XPS8900 Workstation Platform and Lenovo X8800 High-performance Computing Platform**

1. Dell XPS8900
     * System Info
       * System Information
       * Intel® Core™ i7-6700K Processor (8M Cache, up to 4.20 GHz) Product Specifications
     * Testing on Intel Core i7 CPU
       * Source Codes
         * fox_floats_timer_caching_omp_fileIO_benchmark.c
         * serial_matrix_multiplication_benchmark.c
       * Trace Analyzer Test
         * Testing Source Codes and Executive files generated by the Intel Compiler
         * Files generated by Intel® Trace Analyzer
       * Trace Analyzer Results
         * Shortcuts of the analysis result of Intel® Trace Analyzer
2. PKU-HPC
     * Intel Performance Optimization Flow Chat
     * SlurmTemplates
       * testCPUMPISubmit.sh
       * testCPUSubmit_serial.sh
       * testMICMPISubmit.sh
       * testMICSubmit_serial.sh
     * System Info
       * 分区、QOS、账户相关，使用前请详细阅读.one
       * Can Traditional Programming Bridge the Ninja Performance Gap for Parallel Computing Applcations.one
       * Cluster Architecture.docx
       * Intel Xeon Phi processor product brief.one
       * Intel® Xeon Phi™ Processor 7250 (16GB, 1.40 GHz, 68 core) Product Specifications.pdf
       * Intel® Xeon® Processor E5-2697A v4 (40M Cache, 2.60 GHz) Product Specifications.pdf
       * Introduction to Parallel Computing.pdf
       * KNIGHTS LANDING SECONDGENERATION INTEL XEON PHI PRODUCT.one
       * KNL-ISC-2015-Workshop-Keynote.pdf
       * Performance_comparison_Intel_Xeon_Phi_Knights_Landing.pdf
       * System Information.one
       * xeon-phi-processor-product-brief.pdf
     * Testing on Intel Xeon E5 CPUs
       * Source Codes (Source Codes used in the test)
         * testCPUMPISubmit.sh
         * fox_floats_timer_caching_omp_fileIO_benchmark.c
         * testCPUSubmit_serial.sh
         * serial_matrix_multiplication_benchmark.c
       * ProblemScale (Variation Problem's Scale with 16 Processess each Process has 16 Threads)
         * 64*64
         * 128*128
         * 256*256
         * 512*512
         * 1024*1024
         * 2048*2048
         * 4096*4096
         * 8192*8192
         * 16384*16384
       * ProcessScale (Variation Process's Scale with only 1 Threads and Problem's Scale is 8192*8192)
         * 2*2
         * 4*4
         * 8*8
         * 16*16
     * Testing on Intel Xeon Phi KNL MIC
       * Source Codes
         * testMICMPISubmit.sh
         * fox_floats_timer_caching_omp_fileIO_benchmark.c
         * testMICSubmit_serial.sh
         * serial_matrix_multiplication_benchmark.c
       * ProblemScale (Variation Problem's Scale with 64 Processess each Process has 2 Threads)
         * 64*64
         * 128*128
         * 256*256
         * 512*512
         * 1024*1024
         * 2048*2048
         * 4096*4096
         * 8192*8192
       * ProcessScale (Variation Process and Thread's Scale with Problem's Scale is 4096*4096)
         * 4*4*8 (with 4*4 Process each Process 8 Threads)
         * 8*8*2 (with 8*8 Process each Process 2 Threads)