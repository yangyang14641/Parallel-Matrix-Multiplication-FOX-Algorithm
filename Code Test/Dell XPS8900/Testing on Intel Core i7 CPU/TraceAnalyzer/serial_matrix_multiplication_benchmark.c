/* Program name: serial_matrix_multiplication_benchmark.c
 * benchmark serial matrix multiplication with timer
 * 
 */

/* Compiler command: 
 * icc -O3 -xCORE-AVX2 -qopt-report-phase=vec -qopt-report=3 serial_matrix_multiplication_benchmark.c -o serial_matrix_multiplication_benchmark 
 * 
 * Run command:
 * ./serial_matrix_multiplication_benchmark
 */

/* Head files */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>


// define problem scale, matrix row/col size 
#define PROBLEM_SCALE 4096

// define float precision, 4 byte single-precision float or 8 byte double-precision float
#define FLOAT double

// define macro for array access
#define Entry(A,i,j) (*(A + (n*(i) + (j))))    // defination with parameters, Array dereference


// dtime
//
// utility routine to return
// the current wall clock time
//
double dtime()
{
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime, (struct timezone*)0);
    tseconds = (double)(mytime.tv_sec + mytime.tv_usec*1.0e-6);
    return( tseconds );
}


/*********************************************************/
int main(int argc, char* argv[]) {
    FILE             *fp;

    int              i;
    int              j;
    int              k;

    int              n;
    int              content;

    FLOAT           *matrix_A; 
    FLOAT           *matrix_B; 
    FLOAT           *matrix_C;

    double           tstart;
    double           tend;


    // Matrix Generator
    printf("Generate and write matrix A\n");
    fp = fopen("A.dat", "w");           // Generate and print matrix A into a file
    for (i = 0; i < PROBLEM_SCALE; i++)   {                                   
        for (j = 0; j < PROBLEM_SCALE; j++)
        if(i == j){  
            fprintf(fp,"%d ",1);
        }
        else {
            fprintf(fp,"%d ",0);
        }
            
        fprintf(fp,"\n");
    }
    fclose(fp);
    
    printf("Generate and write matrix B\n");
    fp = fopen("B.dat", "w");           // Generate and print matrix B into a file
    for (i = 0; i < PROBLEM_SCALE; i++){                                      
        for (j = 0; j < PROBLEM_SCALE; j++)              
            fprintf(fp,"%d ", (i*PROBLEM_SCALE)+j);
        
        fprintf(fp, "\n");
    }
    fclose(fp);
     

    // Allocate memory and Read Matrices from files
    fp = fopen("A.dat","r");
    n = 0;
    while((content = fgetc(fp)) != EOF)
    {
        //printf("fgetc = %c\n", (char)content);
        if(content != 0x20 && content != 0x0A) n++;
    }
    fclose(fp);
    // printf("We read the order of the matrices from A.dat is\n %d\n", n);
    n = (int) sqrt((double) n); 
    printf("We read the order of the matrices from A.dat is\n %d\n", n);

    matrix_A = (FLOAT*) malloc((n*n) * sizeof(FLOAT));        // Allocate memory for matrix A
    matrix_B = (FLOAT*) malloc((n*n) * sizeof(FLOAT));        // Allocate memory for matrix B
    matrix_C = (FLOAT*) malloc((n*n) * sizeof(FLOAT));        // Allocate memory for matrix C
    
    printf("Read matrix A\n");
    fp = fopen("A.dat", "r");                                 // Read Matrix A from a file
    for (i = 0; i < n; i++)   {                                   
        for (j = 0; j < n; j++) 
            fscanf(fp, "%lf", (matrix_A + (i*n) + j));
    }
    fclose(fp);
    
    printf("Read matrix B\n");
    fp = fopen("B.dat", "r");                                 // Read matrix B from a file
    for (i = 0; i < n; i++){                                      
        for (j = 0; j < n; j++)              
            fscanf(fp, "%lf", (matrix_B + (i*n) + j));
    }
    fclose(fp);

     
    /******************************************************/
    // Serial Matrix Multiply with timer
    /******************************************************/
    for (i = 0; i < n; i++)                                      
        for (j = 0; j < n; j++)              
                Entry(matrix_C,i,j) = 0.0;
    
    printf("Start C = A*B\n");
    tstart = dtime();
    for (i = 0; i < n; i++)                                      
        for (j = 0; j < n; j++)              
            for (k = 0; k < n; k++)
                Entry(matrix_C,i,j) = Entry(matrix_C,i,j)       
                    + Entry(matrix_A,i,k)*Entry(matrix_B,k,j);
    tend = dtime();
    printf("C = A*B finished\n");

    // print timer result in the command line
    printf("Serial Matrix Multiplication Elapsed time:\n %30.20E seconds\n", tend-tstart);
    
    
    // Writing result to a file
    printf("Write result C = A*B to a file C.dat\n");
    fp = fopen("C.dat", "w");                                 // Read matrix B from a file
    for (i = 0; i < PROBLEM_SCALE; i++){                                      
        for (j = 0; j < PROBLEM_SCALE; j++)              
            fprintf(fp,"%lf ", Entry(matrix_C,i,j));
        
        fprintf(fp, "\n");
    }
    fclose(fp);


    return 0;
}
