# **Parallel-Matrix-Multiplication-FOX-Algorithm**
:coffee:Implement of Parallel Matrix Multiplication Methods Using FOX Algorithm on *<span style="color: red">Peking University's High-performance Computing System</span>*

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
    * | Data Mesh     | Mapping       | Process Mesh  |
      | ------------- |:-------------:|:-------------:|
      | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20A_%7B11%7D%20%26%20A_%7B12%7D%20%26%20%5Ccdot%20%26%20A_%7B1p%7D%20%5C%5C%20%0A%20A_%7B21%7D%20%26%20A_%7B22%7D%20%26%20%5Ccdot%20%26%20A_%7B2p%7D%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20A_%7Bp1%7D%20%26%20A_%7Bp2%7D%20%26%20%5Ccdot%20%26%20A_%7Bpp%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)_%7BA_%7Bij%7D%7D">      | <img src="https://tex.s2cms.ru/svg/%5Crightarrow"> | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20P_%7B11%7D%20%26%20P_%7B12%7D%20%26%20%5Ccdot%20%26%20P_%7B1p%7D%20%5C%5C%20%0A%20P_%7B21%7D%20%26%20P_%7B22%7D%20%26%20%5Ccdot%20%26%20P_%7B2p%7D%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20P_%7Bp1%7D%20%26%20P_%7Bp2%7D%20%26%20%5Ccdot%20%26%20P_%7Bpp%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)_%7BP_%7Bij%7D%7D">              |
      | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20B_%7B11%7D%20%26%20B_%7B12%7D%20%26%20%5Ccdot%20%26%20B_%7B1p%7D%20%5C%5C%20%0A%20B_%7B21%7D%20%26%20B_%7B22%7D%20%26%20%5Ccdot%20%26%20B_%7B2p%7D%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20B_%7Bp1%7D%20%26%20B_%7Bp2%7D%20%26%20%5Ccdot%20%26%20B_%7Bpp%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)_%7BB_%7Bij%7D%7D">      | <img src="https://tex.s2cms.ru/svg/%5Crightarrow">      | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20P_%7B11%7D%20%26%20P_%7B12%7D%20%26%20%5Ccdot%20%26%20P_%7B1p%7D%20%5C%5C%20%0A%20P_%7B21%7D%20%26%20P_%7B22%7D%20%26%20%5Ccdot%20%26%20P_%7B2p%7D%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20P_%7Bp1%7D%20%26%20P_%7Bp2%7D%20%26%20%5Ccdot%20%26%20P_%7Bpp%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)_%7BP_%7Bij%7D%7D">             |
      | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20C_%7B11%7D%20%26%20C_%7B12%7D%20%26%20%5Ccdot%20%26%20C_%7B1p%7D%20%5C%5C%20%0A%20C_%7B21%7D%20%26%20C_%7B22%7D%20%26%20%5Ccdot%20%26%20C_%7B2p%7D%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20C_%7Bp1%7D%20%26%20C_%7Bp2%7D%20%26%20%5Ccdot%20%26%20C_%7Bpp%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)_%7BC_%7Bij%7D%7D"> | <img src="https://tex.s2cms.ru/svg/%5Crightarrow">      | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20P_%7B11%7D%20%26%20P_%7B12%7D%20%26%20%5Ccdot%20%26%20P_%7B1p%7D%20%5C%5C%20%0A%20P_%7B21%7D%20%26%20P_%7B22%7D%20%26%20%5Ccdot%20%26%20P_%7B2p%7D%20%5C%5C%20%0A%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%26%20%5Ccdot%20%5C%5C%0A%20P_%7Bp1%7D%20%26%20P_%7Bp2%7D%20%26%20%5Ccdot%20%26%20P_%7Bpp%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)_%7BP_%7Bij%7D%7D">              |

  * *<span style="color: red">Partition may not perfect such that every sub-matrix is a square matrix. Yet, that's not a problem, except load unbalance on each process!</span>*
  
  * Unbalanced Partition
    *  | Unblanced Partition|
       |--------------------|
       | <img src="https://tex.s2cms.ru/svg/%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20c_%7B11%7D%20%26%20%7C%20%26%20c_%7B12%7D%20%5C%5C%20%0A%20%20%20%20%20%20-%20%26%20%7C%20%26%20-%20%5C%5C%0A%20c_%7B21%7D%20%26%20%7C%20%26%20c_%7B22%7D%20%5C%5C%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)%20%0A%3D%0A%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20a_%7B11%7D%20%26%20a_%7B12%7D%20%26%20%7C%20%26a_%7B13%7D%20%26%20a_%7B14%7D%20%26%20a_%7B15%7D%20%5C%5C%20%0A%20-%20%26%20-%20%26%20%7C%20%26%20-%20%26%20-%20%26%20-%20%5C%5C%20%0A%20a_%7B21%7D%20%26%20a_%7B22%7D%20%26%20%7C%20%26a_%7B21%7D%20%26%20a_%7B24%7D%20%26%20a_%7B25%7D%20%5C%5C%20%20%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)%0A%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%20b_%7B11%7D%20%26%20%7C%20%26%20b_%7B12%7D%20%5C%5C%20%0A%20b_%7B21%7D%20%26%20%7C%20%26%20b_%7B22%7D%20%5C%5C%0A%20-%20%26%20%7C%20%26%20-%20%5C%5C%20%0A%20b_%7B31%7D%20%26%20%7C%20%26%20b_%7B32%7D%20%5C%5C%0A%20b_%7B41%7D%20%26%20%7C%20%26%20b_%7B42%7D%20%5C%5C%0A%20b_%7B51%7D%20%26%20%7C%20%26%20b_%7B52%7D%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)"> |
       | <img src="https://tex.s2cms.ru/svg/c_%7B11%7D%20%3D%20%0A%5Cleft%20%5B%0A%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%C2%A0a_%7B11%7D%20%26%20a_%7B12%7D%20%5C%5C%20%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)%0A%C2%A0%0A%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%C2%A0b_%7B11%7D%20%5C%5C%20%0A%C2%A0b_%7B21%7D%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)%0A%5Cright%20%5D%0A%C2%A0%0A%2B%0A%C2%A0%0A%5Cleft%20%5B%0A%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%C2%A0a_%7B13%7D%20%26%20a_%7B14%7D%20%26%20a_%7B15%7D%20%5C%5C%20%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)%0A%C2%A0%0A%5Cleft%20(%0A%5Cbegin%7Bmatrix%7D%0A%C2%A0b_%7B31%7D%20%5C%5C%20%0A%C2%A0b_%7B41%7D%20%5C%5C%0A%C2%A0b_%7B51%7D%0A%5Cend%7Bmatrix%7D%0A%5Cright%20)%0A%5Cright%5D"> |


       <img src="https:">