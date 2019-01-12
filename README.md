# **Parallel-Matrix-Multiplication-FOX-Algorithm**
:coffee:Implement of Parallel Matrix Multiplication Methods Using FOX Algorithm on *<span style="color: red">Peking University's High-performance Computing System</span>*

<center>
<img src="http://logok.org/wp-content/uploads/2014/04/Peking-University-logo.png" width="50%" height="50%" />
</center>
 
 Yes We Code
<center>
<img src="https://octodex.github.com/images/baracktocat.jpg" width="50%" height="50%" />
</center>

## Contents

1. **Reference Documents**
   * Thomas Anastasio, Example of Matrix Multiplication by Fox Method
   * Jaeyoung Choi, A New Parallel Matrix Multiplication Algorithm on Distributed-Memory Concurrent Computers
   * Ned Nedialkov, Communicators and Topologies: Matrix Multiplication Example
2. **Source Codes**
   * C language
   * Fortran
   * [Source Codes' Contents](http://github.com)
3. **Code Tests**
   * Dell XPS8900 
     * Code Test on Dell XPS8900 Workstation (Intel® Core™ i7-6700K Processor)
     * Analyzing MPI Performance Using Intel Trace Analyzer
   * PKU-HPC 
     * Lenovo X8800 Supercomputer Platform
     * Code Performance Tests on X8800 Supercomputer Platform's CPU Node (Intel® Xeon® Processor E5-2697A v4)
     * Code Performance Tests on X8800 Supercomputer Platform's MIC Node (Intel® Xeon Phi™ Processor 7250)
   * [Code Tests' Contents](http://github.com)
4. **Reports**
   * 1801111621_洪瑶_并行程序报告.pdf
   * 并行程序报告.docx
   * 洪瑶_1801111621并行程序设计报告.pptx
   * Parallel FOX Algorithm Project Report.pptx (will be added in the future)
   * Parallel FOX Algorithm Project Report Paper.tex (will be added in the future)
   * Parallel FOX Algorithm Project Report Paper.pdf (will be added in the future)
   * [Reports' Contents](https://github.com/yangyang14641/Parallel-Matrix-Multiplication-FOX-Algorithm/blob/master/Report/CONTENTS.md)
5. **Imagines**
   * FOX.png
   * FOX Stage Whole.JPG
   * FOX Stage Loading Balance.png

## **Brief Introduction to Parallel Matrix Multiplication FOX Algorithm**

### **Basic Concepts**

* 规约计算 (Reduction)
* 拥有者计算原则 (Owner Computing Rule)
* 流水并行(Pipeline Parallelism):
  * 在一个进程上，矩阵计算被划分为P个阶段 (P Supercomputing Steps in a Process)
* 数据并行 (Data Parallelism):
  * 在每个进程上同时计算局部的矩阵乘积 (Local Matrix Multiplications are computing on every processess at the same Computing Step)

### **Serial Matrix Multiplication**

* Mathematical Modeling of Matrix Multiplication
  * <img src="https://tex.s2cms.ru/svg/C_%7Bij%7D%3D%5Csum_%7Bk%3D0%7D%5E%7BK-1%7D%20A_%7Bik%7DB_%7Bkj%7D%3B%20%5Cquad%20(i%3D0%2CN-1)%2C%20%5Cquad%20(j%3D0%2CM-1)" alt="C_{ij}=\sum_{k=0}^{K-1} A_{ik}B_{kj}; \quad (i=0,N-1), \quad (j=0,M-1)" />

* Time Complexity
  * <img src="https://tex.s2cms.ru/svg/O%5Cleft%20(%20N%5E%7B3%7D%20%5Cright%20)" alt="O\left ( N^{3} \right )" />

* Storage Complexity
  * <img src="https://tex.s2cms.ru/svg/O%5Cleft%20(%20N%5E%7B3%7D%20%5Cright%20)" alt="O\left ( N^{3} \right )" />

* Example Implementation in C Language

``` c
for (i = 0; i < n; i++)                                      
        for (j = 0; j < n; j++)              
            for (k = 0; k < n; k++)
                C(i,j) = C(i,j) + A(i,k)*B(k,j);
```

### **Parallel Computing Modeling Design**

1. **Basic Flow**
* Matrix <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA%7D" alt="\mathbf{A}">'s Dimension is <img src="https://tex.s2cms.ru/svg/M%20%5Ctimes%20K" alt="M \times K">, and Matirx <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB%7D" alt="\mathbf{B}">'s Dimension is a <img src="https://tex.s2cms.ru/svg/K%20%5Ctimes%20N" alt="K \times N">.
* Compute Matrix <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC%7D%20%3D%20%5Cmathbf%7BA%7D%5Cmathbf%7BB%7D" alt="\mathbf{C} = \mathbf{A}\mathbf{B}"> in parallel.
* Let <img src="https://tex.s2cms.ru/svg/p%3Dnum(processors)" alt="p=num(processors)"> is the number of processors, and <img src="https://tex.s2cms.ru/svg/q%3D%5Csqrt%7Bp%7D" alt="q=\sqrt{p}"> be an integer such that it devides <img src="https://tex.s2cms.ru/svg/M" alt="M"> and <img src="https://tex.s2cms.ru/svg/N" alt="N">.
* Create a Cartesian topology with process mesh <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}">, and <img src="https://tex.s2cms.ru/svg/i%3D0..q-1" alt="i=0..q-1">, <img src="https://tex.s2cms.ru/svg/j%3D0..q-1" alt="j=0..q-1">.
* Denote <img src="https://tex.s2cms.ru/svg/%5Chat%7BM%7D%20%3D%20%5Cfrac%7BM%7D%7Bq%7D" alt="\hat{M} = \frac{M}{q}">, <img src="https://tex.s2cms.ru/svg/%5Chat%7BK%7D%3D%5Cfrac%7BK%7D%7Bq%7D" alt="\hat{K}=\frac{K}{q}">, <img src="https://tex.s2cms.ru/svg/%5Chat%7BN%7D%3D%5Cfrac%7BN%7D%7Bq%7D" alt="\hat{N}=\frac{N}{q}">.
* Distribute <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA%7D" alt="\mathbf{A}"> and <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB%7D" alt="\mathbf{B}"> by blocks on p processess such that <img src="https://tex.s2cms.ru/svg/A_%7Bij%7D" alt="A_{ij}"> is <img src="https://tex.s2cms.ru/svg/%5Chat%7BM%7D%20%5Ctimes%20%5Chat%7BK%7D" alt="\hat{M} \times \hat{K}"> block and <img src="https://tex.s2cms.ru/svg/B_%7Bij%7D" alt="B_{ij}">  is <img src="https://tex.s2cms.ru/svg/%5Chat%7BK%7D%20%5Ctimes%20%5Chat%7BN%7D" alt="\hat{K} \times \hat{N}"> block, stored on process <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}">.

2. **Details**

* Partitions of Matrices A, B and C. (Index syntax in Mathematical form: start from 1)
  * Matrix A
    * <img src="https://tex.s2cms.ru/svg/A%20%3D%20%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%20%5Cleft(%0A%20%0A%20%20%5Cbegin%7Bmatrix%7D%20%0A%20%20%20%20a_%7B11%7D%20%26%20a_%7B12%7D%20%26%20%5Ccdot%20%26%20a_%7B1%2C%5Cfrac%7BK%7D%7Bq%7D%7D%5C%5C%0A%20%20%20%20a_%7B21%7D%20%26%20a_%7B22%7D%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20%20%20%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20%20%20%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20%20%20%20a_%7B%5Cfrac%7BM%7D%7Bq%7D%2C1%7D%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20a_%7B%5Cfrac%7BM%7D%7Bq%7D%2C%5Cfrac%7BK%7D%7Bq%7D%7D%0A%20%20%5Cend%7Bmatrix%7D%20%5Cright%20)_%7BA_%7B11%7D%7D%20%0A%20%20%20%0A%20%20%20%26%20A_%7B12%7D%20%26%20%5Ccdot%20%26%20A_%7B1q%7D%20%5C%5C%20%0A%20A_%7B21%7D%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20A_%7Bq1%7D%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20A_%7Bqq%7D%20%0A%0A%5Cend%7Bmatrix%7D%20%5Cright%20)_%7BMK%7D" />

  * Matirx B
    * <img src="https://tex.s2cms.ru/svg/B%20%3D%20%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%20%5Cleft(%0A%20%0A%20%20%5Cbegin%7Bmatrix%7D%20%0A%20%20%20%20b_%7B11%7D%20%26%20b_%7B12%7D%20%26%20%5Ccdot%20%26%20b_%7B1%2C%5Cfrac%7BN%7D%7Bq%7D%7D%5C%5C%0A%20%20%20%20b_%7B21%7D%20%26%20b_%7B22%7D%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20%20%20%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20%20%20%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20%20%20%20b_%7B%5Cfrac%7BK%7D%7Bq%7D%2C1%7D%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20b_%7B%5Cfrac%7BK%7D%7Bq%7D%2C%5Cfrac%7BN%7D%7Bq%7D%7D%0A%20%20%5Cend%7Bmatrix%7D%20%5Cright%20)_%7BB_%7B11%7D%7D%20%0A%20%20%20%0A%20%20%20%26%20B_%7B12%7D%20%26%20%5Ccdot%20%26%20B_%7B1q%7D%20%5C%5C%20%0A%20B_%7B21%7D%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20B_%7Bq1%7D%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20B_%7Bqq%7D%20%0A%0A%5Cend%7Bmatrix%7D%20%5Cright%20)_%7BKN%7D" />

  * Matrix C
    * <img src="https://tex.s2cms.ru/svg/C%20%3D%20%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%20%5Cleft(%0A%20%0A%20%20%5Cbegin%7Bmatrix%7D%20%0A%20%20%20%20c_%7B11%7D%20%26%20c_%7B12%7D%20%26%20%5Ccdot%20%26%20c_%7B1%2C%5Cfrac%7BN%7D%7Bq%7D%7D%5C%5C%0A%20%20%20%20c_%7B21%7D%20%26%20c_%7B22%7D%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20%20%20%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20%20%20%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20%20%20%20c_%7B%5Cfrac%7BM%7D%7Bq%7D%2C1%7D%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20c_%7B%5Cfrac%7BM%7D%7Bq%7D%2C%5Cfrac%7BN%7D%7Bq%7D%7D%0A%20%20%5Cend%7Bmatrix%7D%20%5Cright%20)_%7BC_%7B11%7D%7D%20%0A%20%20%20%0A%20%20%20%26%20C_%7B12%7D%20%26%20%5Ccdot%20%26%20C_%7B1q%7D%20%5C%5C%20%0A%20C_%7B21%7D%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20C_%7Bq1%7D%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20C_%7Bqq%7D%20%0A%0A%5Cend%7Bmatrix%7D%20%5Cright%20)_%7BMN%7D" />

* Data Distribution on the 2-D Cartesian Topology Processes Mesh (Index syntax in Mathematical formulars: start from 1)
  * Data Mapping
    * |   Data Mesh   |    Mapping    | Process Mesh  |
      | ------------- |:-------------:|:-------------:|
      | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20A_%7B11%7D%20%26%20A_%7B12%7D%20%26%20%5Ccdot%20%26%20A_%7B1p%7D%20%5C%5C%20%0A%20A_%7B21%7D%20%26%20A_%7B22%7D%20%26%20%5Ccdot%20%26%20A_%7B2p%7D%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20A_%7Bp1%7D%20%26%20A_%7Bp2%7D%20%26%20%5Ccdot%20%26%20A_%7Bpp%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)_%7BA_%7Bij%7D%7D">      | <img src="https://tex.s2cms.ru/svg/%5Crightarrow"> | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20P_%7B11%7D%20%26%20P_%7B12%7D%20%26%20%5Ccdot%20%26%20P_%7B1p%7D%20%5C%5C%20%0A%20P_%7B21%7D%20%26%20P_%7B22%7D%20%26%20%5Ccdot%20%26%20P_%7B2p%7D%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20P_%7Bp1%7D%20%26%20P_%7Bp2%7D%20%26%20%5Ccdot%20%26%20P_%7Bpp%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)_%7BP_%7Bij%7D%7D">              |
      | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20B_%7B11%7D%20%26%20B_%7B12%7D%20%26%20%5Ccdot%20%26%20B_%7B1p%7D%20%5C%5C%20%0A%20B_%7B21%7D%20%26%20B_%7B22%7D%20%26%20%5Ccdot%20%26%20B_%7B2p%7D%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20B_%7Bp1%7D%20%26%20B_%7Bp2%7D%20%26%20%5Ccdot%20%26%20B_%7Bpp%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)_%7BB_%7Bij%7D%7D">      | <img src="https://tex.s2cms.ru/svg/%5Crightarrow">      | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20P_%7B11%7D%20%26%20P_%7B12%7D%20%26%20%5Ccdot%20%26%20P_%7B1p%7D%20%5C%5C%20%0A%20P_%7B21%7D%20%26%20P_%7B22%7D%20%26%20%5Ccdot%20%26%20P_%7B2p%7D%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20P_%7Bp1%7D%20%26%20P_%7Bp2%7D%20%26%20%5Ccdot%20%26%20P_%7Bpp%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)_%7BP_%7Bij%7D%7D">             |
      | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20C_%7B11%7D%20%26%20C_%7B12%7D%20%26%20%5Ccdot%20%26%20C_%7B1p%7D%20%5C%5C%20%0A%20C_%7B21%7D%20%26%20C_%7B22%7D%20%26%20%5Ccdot%20%26%20C_%7B2p%7D%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20C_%7Bp1%7D%20%26%20C_%7Bp2%7D%20%26%20%5Ccdot%20%26%20C_%7Bpp%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)_%7BC_%7Bij%7D%7D"> | <img src="https://tex.s2cms.ru/svg/%5Crightarrow">      | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20P_%7B11%7D%20%26%20P_%7B12%7D%20%26%20%5Ccdot%20%26%20P_%7B1p%7D%20%5C%5C%20%0A%20P_%7B21%7D%20%26%20P_%7B22%7D%20%26%20%5Ccdot%20%26%20P_%7B2p%7D%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20P_%7Bp1%7D%20%26%20P_%7Bp2%7D%20%26%20%5Ccdot%20%26%20P_%7Bpp%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)_%7BP_%7Bij%7D%7D">              |

  * *<span style="color: red">Partition may not perfect such that every sub-matrix is a square matrix. Yet, that's not a problem, except load unbalance on each process!</span>*
  
  * Unbalanced Partition
    *  |        Item         |        Object      |
       |---------------------|--------------------|
       | Data Partition | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20c_%7B11%7D%20%26%20%7C%20%26%20c_%7B12%7D%20%5C%5C%20%0A%20%20%20%20%20%20-%20%26%20%7C%20%26%20-%20%5C%5C%0A%20c_%7B21%7D%20%26%20%7C%20%26%20c_%7B22%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)%20%0A%3D%0A%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20a_%7B11%7D%20%26%20a_%7B12%7D%20%26%20%7C%20%26a_%7B13%7D%20%26%20a_%7B14%7D%20%26%20a_%7B15%7D%20%5C%5C%20%0A%20-%20%26%20-%20%26%20%7C%20%26%20-%20%26%20-%20%26%20-%20%5C%5C%20%0A%20a_%7B21%7D%20%26%20a_%7B22%7D%20%26%20%7C%20%26a_%7B21%7D%20%26%20a_%7B24%7D%20%26%20a_%7B25%7D%20%5C%5C%20%20%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)%0A%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20b_%7B11%7D%20%26%20%7C%20%26%20b_%7B12%7D%20%5C%5C%20%0A%20b_%7B21%7D%20%26%20%7C%20%26%20b_%7B22%7D%20%5C%5C%0A%20-%20%26%20%7C%20%26%20-%20%5C%5C%20%0A%20b_%7B31%7D%20%26%20%7C%20%26%20b_%7B32%7D%20%5C%5C%0A%20b_%7B41%7D%20%26%20%7C%20%26%20b_%7B42%7D%20%5C%5C%0A%20b_%7B51%7D%20%26%20%7C%20%26%20b_%7B52%7D%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)"> |
       | Data Partition | <img src="https://tex.s2cms.ru/svg/c_%7B11%7D%20%3D%20%0A%5Cleft%20%5B%0A%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%C2%A0a_%7B11%7D%20%26%20a_%7B12%7D%20%5C%5C%20%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)%0A%C2%A0%0A%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%C2%A0b_%7B11%7D%20%5C%5C%20%0A%C2%A0b_%7B21%7D%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)%0A%5Cright%20%5D%0A%C2%A0%0A%2B%0A%C2%A0%0A%5Cleft%20%5B%0A%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%C2%A0a_%7B13%7D%20%26%20a_%7B14%7D%20%26%20a_%7B15%7D%20%5C%5C%20%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)%0A%C2%A0%0A%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%C2%A0b_%7B31%7D%20%5C%5C%20%0A%C2%A0b_%7B41%7D%20%5C%5C%0A%C2%A0b_%7B51%7D%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)%0A%5Cright%5D"> |
       | Process Mesh | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20P_%7B11%7D%20%26%20P_%7B12%7D%20%5C%5C%20%0A%20P_%7B21%7D%20%26%20P_%7B22%7D%20%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)"> |

  * Mathematical Modeling of Sub-Matirx Multiplication
    * <img src="https://tex.s2cms.ru/svg/%5Cbegin%7Bequation%7D%0A%20%20%20%5Cbegin%7Baligned%7D%0A%20%20%20%20%20%20%5Cmathbf%7BC_%7Bij%7D%7D%20%26%3D%20%5Csum_%7Bk%3D0%7D%5E%7Bq-1%7D%20%5Cmathbf%7BA_%7Bik%7D%7D%5Cmathbf%7BB_%7Bkj%7D%7D%20%5C%5C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%26%3D%20%5Cmathbf%7BA_%7Bi0%7D%7D%5Cmathbf%7BB_%7B0j%7D%7D%20%2B%5Cmathbf%7B%20A_%7Bi1%7D%7D%5Cmathbf%7BB_%7B1j%7D%7D%20%2B%20%E2%80%A6%20%2B%20%5Cmathbf%7BA_%7Bii-1%7D%7D%5Cmathbf%7BB_%7Bi-1j%7D%7D%20%5Cnewline%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%26%2B%20%5Cmathbf%7BA_%7Bii%7D%7D%5Cmathbf%7BB_%7Bij%7D%7D%20%5C%5C%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%26%2B%20%5Cmathbf%7BA_%7Bi%2Ci%2B1%7D%7D%5Cmathbf%7BB_%7Bi%2B1j%7D%7D%20%2B%20%E2%80%A6%20%2B%20%5Cmathbf%7BA_%7Bi%2Cq-1%7D%7D%5Cmathbf%7BB_%7Bq-1%2Cj%7D%7D%0A%20%20%20%5Cend%7Baligned%7D%0A%5Cend%7Bequation%7D" />

### **Parallel Algorithm Design on BSP**
*<span style="color: violet">Parallelism type: Data parallelism with Pipeline parallelism</span>*

1. Rewrite the formula of Sub-Matirx Multiplication as q−1 Supercomputing Steps
   * | Stage | Mathematical Operation |
     |-------|------------------------|
     |   0   | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%7D%3D%5Cmathbf%7BA_%7Bii%7D%7D%5Cmathbf%7BB_%7Bij%7D%7D" alt="\mathbf{C_{ij}}=\mathbf{A_{ii}}\mathbf{B_{ij}}"> |
     |   1   | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%7D%3D%5Cmathbf%7BC_%7Bij%7D%7D%2B%5Cmathbf%7BA_%7Bii%2B1%7D%7D%5Cmathbf%7BB_%7Bi%2B1j%7D%7D" alt="\mathbf{C_{ij}}=\mathbf{C_{ij}}+\mathbf{A_{ii+1}}\mathbf{B_{i+1j}}"> |
     |   2   | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%7D%3D%5Cmathbf%7BC_%7Bij%7D%7D%2B%5Cmathbf%7BA_%7Bii%2B2%7D%7D%5Cmathbf%7BB_%7Bi%2B2j%7D%7D" alt="\mathbf{C_{ij}}=\mathbf{C_{ij}}+\mathbf{A_{ii+2}}\mathbf{B_{i+2j}}"> |
     |  ...  |        ...         |
     | q-2-i | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%7D%3D%5Cmathbf%7BC_%7Bij%7D%7D%2B%5Cmathbf%7BA_%7Biq-2%7D%7D%5Cmathbf%7BB_%7Bq-2j%7D%7D" alt="\mathbf{C_{ij}}=\mathbf{C_{ij}}+\mathbf{A_{iq-2}}\mathbf{B_{q-2j}}"> |
     | q-1-i | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%7D%3D%5Cmathbf%7BC_%7Bij%7D%7D%2B%5Cmathbf%7BA_%7Biq-1%7D%7D%5Cmathbf%7BB_%7Bq-1j%7D%7D" alt="\mathbf{C_{ij}}=\mathbf{C_{ij}}+\mathbf{A_{iq-1}}\mathbf{B_{q-1j}}"> |
     |  ...  | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%7D%3D%5Cmathbf%7BC_%7Bij%7D%7D%2B%5Cmathbf%7BA_%7Bi1%7D%7D%5Cmathbf%7BB_%7B1j%7D%7D" alt="\mathbf{C_{ij}}=\mathbf{C_{ij}}+\mathbf{A_{i1}}\mathbf{B_{1j}}"> |
     |  ...  | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%7D%3D%5Cmathbf%7BC_%7Bij%7D%7D%2B%5Cmathbf%7BA_%7Bi2%7D%7D%5Cmathbf%7BB_%7B2j%7D%7D" alt="\mathbf{C_{ij}}=\mathbf{C_{ij}}+\mathbf{A_{i2}}\mathbf{B_{2j}}"> |
     |  ...  |        ...         |
     |  q-1  | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%7D%3D%5Cmathbf%7BC_%7Bij%7D%7D%2B%5Cmathbf%7BA_%7Bii-1%7D%7D%5Cmathbf%7BB_%7Bi-1j%7D%7D" alt="\mathbf{C_{ij}}=\mathbf{C_{ij}}+\mathbf{A_{ii-1}}\mathbf{B_{i-1j}}"> |

   * *<span style="color: blue">Data parallelism: Local Matrix Multiplication operation in each processes for each supercomputing step.</span>*
  
2. Parallel Modeling Algorithm Operations on each step:
   * | Stage | Algorithm Operation |
     |-------|---------------------|
     |   0   | 1. Process <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}">  has <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bij%7D%7D" alt="\mathbf{A_{ij}}">, <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bij%7D%7D" alt="\mathbf{B_{ij}}">  but needs <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bii%7D%7D" alt="\mathbf{A_{ii}}"> (for each index <img src="https://tex.s2cms.ru/svg/i" alt="i">) <br/> 2. Process <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BP_%7Bii%7D%7D" alt=\mathbf{P_{ii}}> broadcast <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bii%7D%7D" alt="\mathbf{A_{ii}}"> across process mesh row <img src="https://tex.s2cms.ru/svg/i" alt="i"> <br/> 3. Process <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}"> computes <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DA_%7Bii%7DB_%7Bij%7D%7D" alt="\mathbf{C_{ij}=A_{ii}B_{ij}}"> |
     |   1   | 1. <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}"> has <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bij%7D%7D" alt="\mathbf{A_{ij}}"> and <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bij%7D%7D" alt="\mathbf{B_{ij}}"> but needs <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bii%2B1%7D%7D" alt="\mathbf{A_{ii+1}}"> and <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bi%2B1j%7D%7D" alt="\mathbf{B_{i+1j}}"> <br/> 1.1 Shift the <img src="https://tex.s2cms.ru/svg/j-th" alt="j-th"> block column of <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bij%7D%7D" alt="\mathbf{B_{ij}}"> by one block up (block <img src="https://tex.s2cms.ru/svg/0" alt="0"> goes to block <img src="https://tex.s2cms.ru/svg/q-1" alt="q-1">) (period) <br/> 1.2 <img src="https://tex.s2cms.ru/svg/P_%7Bii%2B1%7D" alt="P_{ii+1}"> broadcast <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bii%2B1%7D%7D" alt="\mathbf{A_{ii+1}}">  across process mesh row <img src="https://tex.s2cms.ru/svg/i" alt="i"> <br/> 2. Process <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}"> Compute <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Bii%2B1%7DB_%7Bi%2B1j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{ii+1}B_{i+1j}}"> |
     |   2   | 1. <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}"> has <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bij%7D%7D" alt="\mathbf{A_{ij}}">  and <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bij%7D%7D" alt="\mathbf{B_{ij}}">  but needs <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bii%2B2%7D%7D" alt="\mathbf{A_{ii+2}}">  and <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bi%2B2j%7D%7D" alt="\mathbf{B_{i+2j}}"> <br/> 1.1 Shift the <img src="https://tex.s2cms.ru/svg/j-th" alt="j-th"> block column of <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bij%7D%7D" alt="\mathbf{B_{ij}}"> by one block up (block <img src="https://tex.s2cms.ru/svg/0" alt="0"> goes to block <img src="https://tex.s2cms.ru/svg/q-1" alt="q−1">) (period) <br/> 1.2 <img src="https://tex.s2cms.ru/svg/P_%7Bii%2B2%7D" alt="P_{ii+2}"> broadcast <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bii%2B2%7D%7D" alt="\mathbf{A_{ii+2}}">  across process mesh row <img src="https://tex.s2cms.ru/svg/i" alt="i"> <br/> 2. Process <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}">  Compute <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Bii%2B2%7DB_%7Bi%2B2j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{ii+2}B_{i+2j}}"> |
     |  ...  |         ...         |
     | q-2-i | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Biq-2%7DB_%7Bq-2j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{iq-2}B_{q-2j}}"> |
     | q-1-i | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Biq-1%7DB_%7Bq-1j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{iq-1}B_{q-1j}}"> |
     |  ...  | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Bi1%7DB_%7B1j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{i1}B_{1j}}"> |
     |  ...  | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Bi2%7DB_%7B2j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{i2}B_{2j}}"> |
     |  ...  |         ...         |
     |  q-1  | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Bii-1%7DB_%7Bi-1j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{ii-1}B_{i-1j}}"> |

   * *<span style="color: blue">Pipe parallelism: The <img src="https://tex.s2cms.ru/svg/(0%20%5Cto%20q-1)" alt="(0 \to q-1)"> Computing Steps for each process <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}">.</span>*
   * <center> <img src="Imagines/FOX.png" width="50%" height="50%" /> </center>

### **Algorithm Analysis**

   <img src="https:">

1. **Algorithm Analysis on each Supercomputing Step**

   * | Stage | Algorithm Operation |Computing and  Communication Analysis|
     |-------|---------------------|---------------------|
     |   0   | 1. Process <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}">  has <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bij%7D%7D" alt="\mathbf{A_{ij}}">, <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bij%7D%7D" alt="\mathbf{B_{ij}}">  but needs <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bii%7D%7D" alt="\mathbf{A_{ii}}"> (for each index <img src="https://tex.s2cms.ru/svg/i" alt="i">) <br/> 2. Process <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BP_%7Bii%7D%7D" alt=\mathbf{P_{ii}}> broadcast <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bii%7D%7D" alt="\mathbf{A_{ii}}"> across process mesh row <img src="https://tex.s2cms.ru/svg/i" alt="i"> <br/> 3. Process <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}"> computes <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DA_%7Bii%7DB_%7Bij%7D%7D" alt="\mathbf{C_{ij}=A_{ii}B_{ij}}"> | Communication in Broadcast Operation: <br/> <img src="https://tex.s2cms.ru/svg/%5Cleft(%20q-1%20%5Cright)%20%20%5Ctimes%20q%20%5Ctimes%20%5Cfrac%7BM%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BK%7D%7Bq%7D" alt="\left( q-1 \right)  \times q \times \frac{M}{q} \times \frac{K}{q}"> <br/> Computing in each process: <br/> <img src="https://tex.s2cms.ru/svg/%5Cfrac%7BM%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BK%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BN%7D%7Bq%7D" alt="\frac{M}{q} \times \frac{K}{q} \times \frac{N}{q}"> <br/> Computing in total: <br/> <img src="https://tex.s2cms.ru/svg/%5Cleft(%20%5Cfrac%7BM%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BK%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BN%7D%7Bq%7D%20%5Cright)%20q%20%5Ctimes%20q" alt="\left( \frac{M}{q} \times \frac{K}{q} \times \frac{N}{q} \right) q \times q"> |
     |   1   | 1. <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}"> has <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bij%7D%7D" alt="\mathbf{A_{ij}}"> and <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bij%7D%7D" alt="\mathbf{B_{ij}}"> but needs <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bii%2B1%7D%7D" alt="\mathbf{A_{ii+1}}"> and <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bi%2B1j%7D%7D" alt="\mathbf{B_{i+1j}}"> <br/> 1.1 Shift the <img src="https://tex.s2cms.ru/svg/j-th" alt="j-th"> block column of <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bij%7D%7D" alt="\mathbf{B_{ij}}"> by one block up (block <img src="https://tex.s2cms.ru/svg/0" alt="0"> goes to block <img src="https://tex.s2cms.ru/svg/q-1" alt="q-1">) (period) <br/> 1.2 <img src="https://tex.s2cms.ru/svg/P_%7Bii%2B1%7D" alt="P_{ii+1}"> broadcast <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bii%2B1%7D%7D" alt="\mathbf{A_{ii+1}}">  across process mesh row <img src="https://tex.s2cms.ru/svg/i" alt="i"> <br/> 2. Process <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}"> Compute <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Bii%2B1%7DB_%7Bi%2B1j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{ii+1}B_{i+1j}}"> | Communication in shift operation: <br/> <img src="https://tex.s2cms.ru/svg/%5Cleft(%20q%20%5Ctimes%20q%20%5Cright)%20%5Ctimes%20%5Cleft(%20%5Cfrac%7BK%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BN%7D%7Bq%7D%20%5Cright)" alt="\left( q \times q \right) \times \left( \frac{K}{q} \times \frac{N}{q} \right)"> <br/> Communication in broadcast operation: <br/> <img src="https://tex.s2cms.ru/svg/%5Cleft%5B%5Cleft(%20q-1%20%5Cright)%20%5Ctimes%20q%5Cright%5D%20%5Ctimes%20%5Cleft(%20%5Cfrac%7BM%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BK%7D%7Bq%7D%5Cright)" alt="\left[\left( q-1 \right) \times q\right] \times \left( \frac{M}{q} \times \frac{K}{q}\right)"> <br/> Communication in total: <br/> <img src="https://tex.s2cms.ru/svg/%5Cleft(%20q%20%5Ctimes%20q%20%5Cright)%20%5Ctimes%20%5Cleft(%20%5Cfrac%7BK%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BN%7D%7Bq%7D%5Cright)%20%2B%20%5Cleft%5B%20%5Cleft(%20q-1%20%5Cright)%5Ctimes%20q%20%5Cright%5D%20%5Ctimes%20%5Cleft(%20%5Cfrac%7BM%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BK%7D%7Bq%7D%5Cright)" alt="\left( q \times q \right) \times \left( \frac{K}{q} \times \frac{N}{q}\right) + \left[ \left( q-1 \right)\times q \right] \times \left( \frac{M}{q} \times \frac{K}{q}\right)"> <br/> Computing in each process: <br/> <img src="https://tex.s2cms.ru/svg/%5Cfrac%7BM%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BK%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BN%7D%7Bq%7D" alt="\frac{M}{q} \times \frac{K}{q} \times \frac{N}{q}"> <br/> Computing in total: <br/> <img src="https://tex.s2cms.ru/svg/%5Cleft(%20%5Cfrac%7BM%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BK%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BN%7D%7Bq%7D%20%5Cright)%20%5Ctimes%20q%20%5Ctimes%20q" alt="\left( \frac{M}{q} \times \frac{K}{q} \times \frac{N}{q} \right) \times q \times q"> |
     |   2   | 1. <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}"> has <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bij%7D%7D" alt="\mathbf{A_{ij}}">  and <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bij%7D%7D" alt="\mathbf{B_{ij}}">  but needs <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bii%2B2%7D%7D" alt="\mathbf{A_{ii+2}}">  and <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bi%2B2j%7D%7D" alt="\mathbf{B_{i+2j}}"> <br/> 1.1 Shift the <img src="https://tex.s2cms.ru/svg/j-th" alt="j-th"> block column of <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BB_%7Bij%7D%7D" alt="\mathbf{B_{ij}}"> by one block up (block <img src="https://tex.s2cms.ru/svg/0" alt="0"> goes to block <img src="https://tex.s2cms.ru/svg/q-1" alt="q−1">) (period) <br/> 1.2 <img src="https://tex.s2cms.ru/svg/P_%7Bii%2B2%7D" alt="P_{ii+2}"> broadcast <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BA_%7Bii%2B2%7D%7D" alt="\mathbf{A_{ii+2}}">  across process mesh row <img src="https://tex.s2cms.ru/svg/i" alt="i"> <br/> 2. Process <img src="https://tex.s2cms.ru/svg/P_%7Bij%7D" alt="P_{ij}">  Compute <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Bii%2B2%7DB_%7Bi%2B2j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{ii+2}B_{i+2j}}"> | Communication in shift operation: <br/> <img src="https:" alt="\left( q \times q \right) \times \left( \frac{K}{q} \times \frac{N}{q} \right)"> <br/> Communication in broadcast operation: <br/> <img src="https://tex.s2cms.ru/svg/%5Cleft%5B%5Cleft(%20q-1%20%5Cright)%20%5Ctimes%20q%20%5Cright%5D%20%5Ctimes%20%5Cleft(%20%5Cfrac%7BM%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BK%7D%7Bq%7D%20%5Cright)" alt="\left[\left( q-1 \right) \times q \right] \times \left( \frac{M}{q} \times \frac{K}{q} \right)"> <br/> Communication in total: <br/> <img src="https://tex.s2cms.ru/svg/%5Cleft%5B%5Cleft(%20q-1%20%5Cright)%20%5Ctimes%20q%20%5Cright%5D%20%5Ctimes%20%5Cleft(%20%5Cfrac%7BM%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BK%7D%7Bq%7D%5Cright)" alt="\left[\left( q-1 \right) \times q \right] \times \left( \frac{M}{q} \times \frac{K}{q}\right)"> <br/> Computing in each process: <br/> <img src="https://tex.s2cms.ru/svg/%5Cfrac%7BM%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BK%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BN%7D%7Bq%7D" alt="\frac{M}{q} \times \frac{K}{q} \times \frac{N}{q}"> <br/> Computing in total:<br/><img src="https://tex.s2cms.ru/svg/%5Cleft(%5Cfrac%7BM%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BK%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BN%7D%7Bq%7D%5Cright)%20%5Ctimes%20q%20%5Ctimes%20q" alt="\left(\frac{M}{q} \times \frac{K}{q} \times \frac{N}{q}\right) \times q \times q"> |
     |  ...  |         ...         |  |
     | q-2-i | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Biq-2%7DB_%7Bq-2j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{iq-2}B_{q-2j}}"> |  |
     | q-1-i | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Biq-1%7DB_%7Bq-1j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{iq-1}B_{q-1j}}"> |  |
     |  ...  | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Bi1%7DB_%7B1j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{i1}B_{1j}}"> |  |
     |  ...  | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Bi2%7DB_%7B2j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{i2}B_{2j}}"> |  |
     |  ...  |         ...         |  |
     |  q-1  | <img src="https://tex.s2cms.ru/svg/%5Cmathbf%7BC_%7Bij%7D%3DC_%7Bij%7D%2BA_%7Bii-1%7DB_%7Bi-1j%7D%7D" alt="\mathbf{C_{ij}=C_{ij}+A_{ii-1}B_{i-1j}}"> |  |

2. **Communication in total**
  * <img src="https://tex.s2cms.ru/svg/%5Cbegin%7Balign*%7D%0AComm%20%26%3D%20%5Cleft%5B%20(q-1)q%20%5Ctimes%20%5Cfrac%7BM%20%5Ctimes%20K%7D%7Bq%5E%7B2%7D%7D%20%5Ctimes%20q%20%5Cright%5D%20%5C%5C%0A%26%2B%20%5Cleft%5B%20(q%20%5Ctimes%20q)%20%5Ctimes%20%5Cfrac%7BK%20%5Ctimes%20N%7D%7Bq%5E2%7D%20%5Ctimes(q-1)%20%5Cright%5D%20%5C%5C%0A%26%3D%20%5Cleft(%20M%20%5Ctimes%20K%20%2B%20K%20%5Ctimes%20N%20%5Cright)%20%5Ctimes%20(q-1)%20%0A%5Cend%7Balign*%7D">

3. **Computing in total**
  * <img src="https://tex.s2cms.ru/svg/%5Cbegin%7Balign*%7D%0AComput%20%26%3D%20%20%5Cleft%5B%20%5Cleft(%20%5Cfrac%7BM%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BK%7D%7Bq%7D%20%5Ctimes%20%5Cfrac%7BN%7D%7Bq%7D%20%5Cright)%20%5Ctimes%20%5Cleft(%20q%20%5Ctimes%20q%5Cright)%20%5Cright%5D%20%5Ctimes%20q%20%5C%5C%0A%26%3D%20M%20%5Ctimes%20K%20%5Ctimes%20N%0A%5Cend%7Balign*%7D">

### **FOX Kernel in the Parallel MPI-C Program**

   * ``` c 
         n_bar = n/grid->q;
         Set_to_zero(local_C);

         source = (grid->my_row + 1) % grid->q;
         dest = (grid->my_row + grid->q - 1) % grid->q;

         temp_A = Local_matrix_allocate(n_bar);

         for (stage = 0; stage < grid->q; stage++) {
             bcast_root = (grid->my_row + stage) % grid->q;
             if (bcast_root == grid->my_col) {
               MPI_Bcast(local_A, 1, local_matrix_mpi_t,
                         bcast_root, grid->row_comm);
               Local_matrix_multiply(local_A, local_B,local_C);
             } else {
               MPI_Bcast(temp_A, 1, local_matrix_mpi_t,
                         bcast_root, grid->row_comm);
               Local_matrix_multiply(temp_A, local_B,local_C);
             }
             MPI_Sendrecv_replace(local_B, 1, local_matrix_mpi_t,
                                  dest, 0, source, 0, grid->col_comm, &status);
          }
     ```

## **Analysis**
  * <center> <img src="Imagines/FOX Stage Whole.JPG" width="50%" height="50%" /> </center>
  * <center> <img src="Imagines/FOX Stage Loading Balance.png" width="50%" height="50%" /> </center>

## **Warranty** 
**Maybe, there are many mistakes in the both documents and Codes, because of the limitation of our knowledge and strength. As a result: THESE DOCUMENTS AND CODES ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.
I MAKE NO WARRANTIES, EXPRESS OR IMPLIED, THAT THEY ARE FREE OF ERROR.**

## **Copyright**
**You can use and copy these works for any academic purpose, Except just copy to finish your homework or republish these works without proper declare their original author.**