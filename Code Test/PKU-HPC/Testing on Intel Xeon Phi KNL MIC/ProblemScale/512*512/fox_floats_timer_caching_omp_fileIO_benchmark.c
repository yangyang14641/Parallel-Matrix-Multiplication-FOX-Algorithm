/* fox_floats_timer_caching_omp_fileIO_benchmark.c -- uses Fox's algorithm to multiply two square matrices
 * 
 * Implementation of parallel matrix multiplication:
 * LaTeX: $C_{i,j} = \sum_{k} A_{i,k}B_{k,j}$
 * 
 * Input:
 *     Input Matrix file name: A.dat, B.dat
 * 
 * Output:
 *     Output Matrix file name: C.dat
 *     Output Sub-matrices file name: SubMatrices.dat
 *
 * Notes:  
 *     1.  Assumes the number of processes is a perfect square
 *     2.  The array member of the matrices is statically allocated
 *
 * See Chap 7, pp. 113 & ff and pp. 125 & ff in PPMPI
 */


/* Compiler command: 
 * mpiicc -O3 -qopenmp -qopt-report-phase=vec -qopt-report=3 fox_floats_timer_caching_omp_fileIO_benchmark.c 
 * -o fox_floats_timer_caching_omp_fileIO_benchmark 
 * 
 * Run command:
 * mpirun -n -4 ./fox_floats_timer_caching_omp
 */


/* Head files */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>


// define problem scale, matrix row/col size 
#define PROBLEM_SCALE 512

// define whether or not Print Matices in the Command Line
#define PRINT_A 0
#define PRINT_B 0
#define PRINT_C 0

#define PRINT_LOCAL_A 0
#define PRINT_LOCAL_B 0
#define PRINT_LOCAL_C 0

// define float precision, 4 byte single-precision float or 8 byte double-precision float
#define FLOAT double
#define FLOAT_MPI MPI_DOUBLE

// Define threads speed-up affnity in the computing
#define NUM_THREADS 2
// Define threads affinity "scatter" or "compact" 
#define AFFINITY "KMP_AFFINITY = compact"


/* Type define structure of process grid */
typedef struct {
    int       p;             /* Total number of processes    */
    MPI_Comm  comm;          /* Communicator for entire grid */
    MPI_Comm  row_comm;      /* Communicator for my row      */
    MPI_Comm  col_comm;      /* Communicator for my col      */
    int       q;             /* Order of grid                */
    int       my_row;        /* My row number                */
    int       my_col;        /* My column number             */
    int       my_rank;       /* My rank in the grid comm     */
} GRID_INFO_T;             


/* Type define structure of local matrix */
#define MAX 2097152  // Maximum number of elements in the array that store the local matrix (2^21)
typedef struct {
    int     n_bar;
    #define Order(A) ((A)->n_bar)                                        // defination with parameters
    FLOAT  entries[MAX];
    #define Entry(A,i,j) (*(((A)->entries) + ((A)->n_bar)*(i) + (j)))    // defination with parameters, Array dereference
} LOCAL_MATRIX_T;

/* Function Declarations */
LOCAL_MATRIX_T*  Local_matrix_allocate(int n_bar);
void             Free_local_matrix(LOCAL_MATRIX_T** local_A);

void             Read_matrix_A(char* prompt, LOCAL_MATRIX_T* local_A, 
                     GRID_INFO_T* grid, int n);                          // Read matrix A from a file
void             Read_matrix_B(char* prompt, LOCAL_MATRIX_T* local_B,    // for continuous memory access, local A(i,k)*B(k,j) = A(i,k)*B^{T}(j,k)
                     GRID_INFO_T* grid, int n);                          // Read matrix B from a file

void             Print_matrix_A(char* title, LOCAL_MATRIX_T* local_A,     
                     GRID_INFO_T* grid, int n);                          // Print matrix A in the command line
void             Print_matrix_B(char* title, LOCAL_MATRIX_T* local_B,    // Speical print function for local matrix B^{T}(j,k)
                     GRID_INFO_T* grid, int n);                          // Print matrix B in the command line
void             Print_matrix_C(char* title, LOCAL_MATRIX_T* local_C,     
                     GRID_INFO_T* grid, int n);                          // Print matrix C in the command line

void             Set_to_zero(LOCAL_MATRIX_T* local_A);
void             Local_matrix_multiply(LOCAL_MATRIX_T* local_A,
                     LOCAL_MATRIX_T* local_B, LOCAL_MATRIX_T* local_C);

void             Build_matrix_type(LOCAL_MATRIX_T* local_A);
MPI_Datatype     local_matrix_mpi_t;    

LOCAL_MATRIX_T*  temp_mat;       // global LOCAL_MATRIX_T* type pointer

void             Print_local_matrices_A(char* title, LOCAL_MATRIX_T* local_A, 
                     GRID_INFO_T* grid);
void             Print_local_matrices_B(char* title, LOCAL_MATRIX_T* local_B, // Speical print function for local matrix B^{T}(j,k)
                     GRID_INFO_T* grid);
void             Print_local_matrices_C(char* title, LOCAL_MATRIX_T* local_B, 
                     GRID_INFO_T* grid);

void             Write_matrix_C(char* title, LOCAL_MATRIX_T* local_C, 
                     GRID_INFO_T* grid, int n);                               // Write matrix multiplication to a file

void             Write_local_matrices_A(char* title, LOCAL_MATRIX_T* local_A, 
                     GRID_INFO_T* grid);                                      // Write local matrix A to a file
void             Write_local_matrices_B(char* title, LOCAL_MATRIX_T* local_B, // Speical print function for local matrix B^{T}(j,k)
                     GRID_INFO_T* grid);                                      // Write local matrix B to a file
void             Write_local_matrices_C(char* title, LOCAL_MATRIX_T* local_A, 
                     GRID_INFO_T* grid);                                      // Write local matrix C to a file



/*********************************************************/
main(int argc, char* argv[]) {
    FILE             *fp;
    int              p;
    int              my_rank;
    GRID_INFO_T      grid;
    LOCAL_MATRIX_T*  local_A;
    LOCAL_MATRIX_T*  local_B;
    LOCAL_MATRIX_T*  local_C;
    int              n;
    int              n_bar;
    double           timer_start;
    double           timer_end;
    int              content;
    
    int              i;
    int              j;

    void Setup_grid(GRID_INFO_T*  grid);
    void Fox(int n, GRID_INFO_T* grid, LOCAL_MATRIX_T* local_A,
             LOCAL_MATRIX_T* local_B, LOCAL_MATRIX_T* local_C);
    

    // Matrix Generator
    fp = fopen("A.dat", "w");           // Generate and print matrix A into a file
    for (i = 0; i < PROBLEM_SCALE; i++)   {                                   
        for (j = 0; j < PROBLEM_SCALE; j++)
        if(i == j){  
            fprintf(fp,"%d ", 1);
        }
        else {
            fprintf(fp,"%d ", 0);
        }
            
        fprintf(fp,"\n");
    }
    fclose(fp);
    
    fp = fopen("B.dat", "w");           // Generate and print matrix B into a file
    for (i = 0; i < PROBLEM_SCALE; i++){                                      
        for (j = 0; j < PROBLEM_SCALE; j++)              
            fprintf(fp,"%d ", (i*PROBLEM_SCALE)+j);
        
        fprintf(fp, "\n");
    }
    fclose(fp);


    // SPMD Mode start from here (Processess fork from here) 
    MPI_Init(&argc, &argv);                              // MPI initializing 
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);             // Get my process id in the MPI communicator 

    // Initial OpenMP Environment
    omp_set_num_threads(NUM_THREADS);
    kmp_set_defaults(AFFINITY);

    Setup_grid(&grid);                                   // Set up Processess grid 
    if (my_rank == 0) {
        fp = fopen("A.dat","r");
        n = 0;
        while((content = fgetc(fp)) != EOF)
        {
            //printf("fgetc = %d\n", content);
            if(content != 0x20 && content != 0x0A) n++;
        }
        fclose(fp);
        n = (int) sqrt((double) n); 
        printf("We read the order of the matrices from A.dat is\n %d\n", n);
        // while(fgetc(fp) != EOF) n++;
        // printf("What's the order of the matrices?\n");
        // scanf("%d", &n);                                 // Overall Matrix's Order 
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);        // MPI broadcast the overall matrix's order 
    n_bar = n/grid.q;                                    // \bar n is the local matrix's order 

    local_A = Local_matrix_allocate(n_bar);              // Allocate local matrix A 
    Order(local_A) = n_bar;                              // Local matrix A's order 
    Read_matrix_A("Read A from A.dat", local_A, &grid, n);  // Read local matrices A from process 0 by using stdin, and send them to each process (Procedure)
    if (PRINT_A == 1)
        Print_matrix_A("We read A =", local_A, &grid, n);// Print local matrices A from process 0 by using stdout, and send them to each process (Procedure)

    local_B = Local_matrix_allocate(n_bar);              // Allocate local matrix 
    Order(local_B) = n_bar;                              // Local matrix B's order 
    Read_matrix_B("Read B from B.dat", local_B, &grid, n);         // Read local matrix B as it's local transpose from process 0 by using stdin, and send them to each process (Procedure)
    if (PRINT_B == 1)
        Print_matrix_B("We read B =", local_B, &grid, n);// Print local matrix B as it's local transpose from process 0 by using stdout, and send them to each process (Procedure)

    Build_matrix_type(local_A);                          // Buid local_A's MPI matrix data type
    temp_mat = Local_matrix_allocate(n_bar);             // Allocate temporary matrix of order n $\time$ n
                                              
    local_C = Local_matrix_allocate(n_bar);              // Allocate matrix local_C
    Order(local_C) = n_bar;                              // Set matrix local_C's order 
    
    MPI_Barrier(MPI_COMM_WORLD);                         // Set the MPI process barrier
    timer_start = MPI_Wtime();                           // Get the MPI wall time
    Fox(n, &grid, local_A, local_B, local_C);            // FOX parallel matrix multiplication Algorithm implement function 
    timer_end = MPI_Wtime();                             // Get the MPI wall time
    MPI_Barrier(MPI_COMM_WORLD);                         // Set the MPI process barrier
    
    Write_matrix_C("Write C into the C.dat", local_C, &grid, n); // Print matrix local_C (parallel matrix multiplication result)
    if (PRINT_C == 1)
        Print_matrix_C("The product is", local_C, &grid, n); // Print matrix local_C (parallel matrix multiplication result) 

    Write_local_matrices_A("Write split of local matrix A into local_A.dat", 
                         local_A, &grid);                // Write local matrix A into file
    if (PRINT_LOCAL_A == 1)
        Print_local_matrices_A("Split of local matrix A", 
                         local_A, &grid);                // Print matrix A split in processess

    Write_local_matrices_B("Write split of local matrix B into local_B.dat", 
                         local_B, &grid);                // Write local matrix B into file, special for row-major storage    
    if (PRINT_LOCAL_B == 1)
        Print_local_matrices_B("Split of local matrix B", 
                         local_B, &grid);                // Print matrix B split in processess, special for row-major storage

    Write_local_matrices_C("Write split of local matrix C into local_C.dat", 
                         local_C, &grid);                // Print matrix C split in processess
    if (PRINT_LOCAL_C == 1)
        Print_local_matrices_C("Split of local matrix C", 
                         local_C, &grid);                // Print matrix C split in processess

    Free_local_matrix(&local_A);                         // Free local matrix local_A
    Free_local_matrix(&local_B);                         // Free local matrix local_B
    Free_local_matrix(&local_C);                         // Free local matrix local_C
    
    if(my_rank == 0)  
        printf("Parallel Fox Matrix Multiplication Elapsed time:\n %30.20E seconds\n", timer_end-timer_start);

    MPI_Finalize();                                      // MPI finalize, processes join and resource recycle

}  /* main */


/*********************************************************/
void Setup_grid(
         GRID_INFO_T*  grid  /* out */) {
    int old_rank;
    int dimensions[2];
    int wrap_around[2];
    int coordinates[2];
    int free_coords[2];

    /* Set up Global Grid Information */
    MPI_Comm_size(MPI_COMM_WORLD, &(grid->p));
    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);

    /* We assume p is a perfect square */     // but what if it's not a perfect square 
    grid->q = (int) sqrt((double) grid->p); 
    dimensions[0] = dimensions[1] = grid->q; 

    /* We want a circular shift in second dimension. */ 
    /* Don't care about first                        */ 
    wrap_around[0] = wrap_around[1] = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, 
        wrap_around, 1, &(grid->comm));
    MPI_Comm_rank(grid->comm, &(grid->my_rank));
    MPI_Cart_coords(grid->comm, grid->my_rank, 2, 
        coordinates);
    grid->my_row = coordinates[0];
    grid->my_col = coordinates[1];

    /* Set up row communicators */
    free_coords[0] = 0; 
    free_coords[1] = 1;
    MPI_Cart_sub(grid->comm, free_coords, 
        &(grid->row_comm));

    /* Set up column communicators */
    free_coords[0] = 1; 
    free_coords[1] = 0;
    MPI_Cart_sub(grid->comm, free_coords, 
        &(grid->col_comm));
} /* Setup_grid */


/*********************************************************/
void Fox(
        int              n         /* in  */, 
        GRID_INFO_T*     grid      /* in  */, 
        LOCAL_MATRIX_T*  local_A   /* in  */,
        LOCAL_MATRIX_T*  local_B   /* in  */,
        LOCAL_MATRIX_T*  local_C   /* out */) {

    LOCAL_MATRIX_T*  temp_A; /* Storage for the sub-    */
                             /* matrix of A used during */ 
                             /* the current stage       */
    int              stage;
    int              bcast_root;
    int              n_bar;  /* n/sqrt(p)               */
    int              source;
    int              dest;
    MPI_Status       status;
    
    n_bar = n/grid->q;
    Set_to_zero(local_C);

    /* Calculate addresses for row circular shift of B */  
    source = (grid->my_row + 1) % grid->q;
    dest = (grid->my_row + grid->q - 1) % grid->q;

    /* Set aside storage for the broadcast block of A */
    temp_A = Local_matrix_allocate(n_bar);

    for (stage = 0; stage < grid->q; stage++) {
        bcast_root = (grid->my_row + stage) % grid->q;
        if (bcast_root == grid->my_col) {                       // Process P_{ii} broadcast A_{ii} in process gird's row commnunicator
            MPI_Bcast(local_A, 1, local_matrix_mpi_t,
                bcast_root, grid->row_comm);
            Local_matrix_multiply(local_A, local_B, 
                local_C);
        } else {                                                // temp_A is a buffer for process P_{ij} to store A_{ij}
            MPI_Bcast(temp_A, 1, local_matrix_mpi_t,
                bcast_root, grid->row_comm);
            Local_matrix_multiply(temp_A, local_B, 
                local_C);
        }
        MPI_Sendrecv_replace(local_B, 1, local_matrix_mpi_t,     // MPI send and receive with single buffer
            dest, 0, source, 0, grid->col_comm, &status);        // Circular shift of process grid B's row, after local multiplication operation
    } /* for */
    
} /* Fox */


/*********************************************************/
LOCAL_MATRIX_T* Local_matrix_allocate(int local_order) {
    LOCAL_MATRIX_T* temp;
  
    temp = (LOCAL_MATRIX_T*) malloc(sizeof(LOCAL_MATRIX_T));
    return temp;
}  /* Local_matrix_allocate */


/*********************************************************/
void Free_local_matrix(
         LOCAL_MATRIX_T** local_A_ptr  /* in/out */) {
    free(*local_A_ptr);
}  /* Free_local_matrix */


/*********************************************************/
/* Read and distribute matrix for matrix A:  
 *     foreach global row of the matrix,
 *         foreach grid column 
 *             read a block of n_bar floats on process 0
 *             and send them to the appropriate process.
 */
void Read_matrix_A(
         char*            prompt   /* in  */, 
         LOCAL_MATRIX_T*  local_A  /* out */,
         GRID_INFO_T*     grid     /* in  */,
         int              n        /* in  */) {
    
    FILE *fp;
    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        dest;
    int        coords[2];
    FLOAT*     temp;
    MPI_Status status;
    
    if (grid->my_rank == 0) {  // Process 0 read matrix input from stdin and send them to other processess
        fp = fopen("A.dat","r");
        temp = (FLOAT*) malloc(Order(local_A)*sizeof(FLOAT));
        printf("%s\n", prompt);
        fflush(stdout);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Order(local_A);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &dest);
                if (dest == 0) {
                    for (mat_col = 0; mat_col < Order(local_A); mat_col++)
                        fscanf(fp, "%lf", 
                           (local_A->entries)+mat_row*Order(local_A)+mat_col);
                        /* scanf("%lf", 
                           (local_A->entries)+mat_row*Order(local_A)+mat_col);
                        */
                } else {
                    for(mat_col = 0; mat_col < Order(local_A); mat_col++)
                        fscanf(fp,"%lf", temp + mat_col);
                        // scanf("%lf", temp + mat_col);
                    MPI_Send(temp, Order(local_A), FLOAT_MPI, dest, 0,
                        grid->comm);
                }
            }
        }
        free(temp);
        fclose(fp);
    } else {  // Other processess receive matrix from process 0
        for (mat_row = 0; mat_row < Order(local_A); mat_row++) 
            MPI_Recv(&Entry(local_A, mat_row, 0), Order(local_A), 
                FLOAT_MPI, 0, 0, grid->comm, &status);
    }
                     
}  /* Read_matrix */


/*********************************************************/
/* Read and distribute matrix for local matrix B's transpose:  
 *     foreach global row of the matrix,
 *         foreach grid column 
 *             read a block of n_bar floats on process 0
 *             and send them to the appropriate process.
 */

void Read_matrix_B(
         char*            prompt   /* in  */, 
         LOCAL_MATRIX_T*  local_B  /* out */,
         GRID_INFO_T*     grid     /* in  */,
         int              n        /* in  */) {

    FILE       *fp;
    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        dest;
    int        coords[2];
    FLOAT      *temp;
    MPI_Status status;
    
    if (grid->my_rank == 0) {  // Process 0 read matrix input from stdin and send them to other processess
        fp = fopen("B.dat","r");
        temp = (FLOAT*) malloc(Order(local_B)*sizeof(FLOAT));
        printf("%s\n", prompt);
        fflush(stdout);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Order(local_B);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &dest);
                if (dest == 0) {                                                    // process 0 (local)
                    for (mat_col = 0; mat_col < Order(local_B); mat_col++)
                        fscanf(fp, "%lf", 
                          (local_B->entries)+mat_col*Order(local_B)+mat_row);       // switch rows and colums in local_B, for column major storage
                        /* scanf("%lf", 
                          (local_B->entries)+mat_col*Order(local_B)+mat_row);       // switch rows and colums in local_B, for column major storage
                        */
                        /* scanf("%lf", 
                          (local_A->entries)+mat_row*Order(local_A)+mat_col); */       
                } else {
                    for(mat_col = 0; mat_col < Order(local_B); mat_col++)
                        fscanf(fp, "%lf", temp + mat_col);
                        // scanf("%lf", temp + mat_col);
                    MPI_Send(temp, Order(local_B), FLOAT_MPI, dest, 0,
                        grid->comm);
                }
            }
        }
        free(temp);
        fclose(fp);
    } else {  // Other processess receive matrix from process 0
        temp = (FLOAT*) malloc(Order(local_B)*sizeof(FLOAT));               // switch rows and colums in local_B, for column major storage
        for (mat_col = 0; mat_col < Order(local_B); mat_col++) { 
            MPI_Recv(temp, Order(local_B), 
                FLOAT_MPI, 0, 0, grid->comm, &status);                      // switch rows and colums in local_B, for column major storage
            for(mat_row = 0; mat_row < Order(local_B); mat_row++)
                  Entry(local_B, mat_row, mat_col) = *(temp + mat_row);       // switch rows and colums in local_B, for column major storage

            /* MPI_Recv(&Entry(local_A, mat_row, 0), Order(local_A), 
                FLOAT_MPI, 0, 0, grid->comm, &status); */
        }
        free(temp);
    }
                     
}  /* Read_matrix_B */


/*********************************************************/
/* Recive and Print Matrix A:  
 *     foreach global row of the matrix,
 *         foreach grid column 
 *             send n_bar floats to process 0 from each other process
 *             receive a block of n_bar floats on process 0 from other processes and print them 
 */
void Print_matrix_A(
         char*            title    /* in  */,  
         LOCAL_MATRIX_T*  local_A  /* out */,
         GRID_INFO_T*     grid     /* in  */,
         int              n        /* in  */) {

    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        source;
    int        coords[2];
    FLOAT*     temp;
    MPI_Status status;

    if (grid->my_rank == 0) {
        temp = (FLOAT*) malloc(Order(local_A)*sizeof(FLOAT));
        printf("%s\n", title);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Order(local_A);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &source);
                if (source == 0) {
                    for(mat_col = 0; mat_col < Order(local_A); mat_col++)
                        printf("%20.15E ", Entry(local_A, mat_row, mat_col));
                } else {
                    MPI_Recv(temp, Order(local_A), FLOAT_MPI, source, 0,
                        grid->comm, &status);
                    for(mat_col = 0; mat_col < Order(local_A); mat_col++)
                        printf("%20.15E ", temp[mat_col]);
                }
            }
            printf("\n");
        }
        free(temp);
    } else {
        for (mat_row = 0; mat_row < Order(local_A); mat_row++) 
            MPI_Send(&Entry(local_A, mat_row, 0), Order(local_A), 
                FLOAT_MPI, 0, 0, grid->comm);
    }
                     
}  /* Print_matrix_A */


/*********************************************************/
/* Recive and Print Matrix for local matrix B's transpose:  
 *     foreach global row of the matrix,
 *         foreach grid column 
 *             send n_bar floats to process 0 from each other process
 *             receive a block of n_bar floats on process 0 from other processes and print them 
 */
void Print_matrix_B(
         char*            title    /* in  */,  
         LOCAL_MATRIX_T*  local_B  /* out */,
         GRID_INFO_T*     grid     /* in  */,
         int              n        /* in  */) {
    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        source;
    int        coords[2];
    FLOAT*     temp;
    MPI_Status status;

    if (grid->my_rank == 0) {
        temp = (FLOAT*) malloc(Order(local_B)*sizeof(FLOAT));
        printf("%s\n", title);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Order(local_B);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &source);
                if (source == 0) {
                    for(mat_col = 0; mat_col < Order(local_B); mat_col++)
                        printf("%20.15E ", Entry(local_B, mat_col, mat_row));      // switch rows and colums in local_B, for column major storage
                        // printf("%20.15E ", Entry(local_A, mat_row, mat_col));
                } else {
                    MPI_Recv(temp, Order(local_B), FLOAT_MPI, source, 0,
                        grid->comm, &status);
                    for(mat_col = 0; mat_col < Order(local_B); mat_col++)
                        printf("%20.15E ", temp[mat_col]);
                }
            }
            printf("\n");
        }
        free(temp);
    } else {
        temp = (FLOAT*) malloc(Order(local_B)*sizeof(FLOAT));
        for (mat_col = 0; mat_col < Order(local_B); mat_col++) { 
            for(mat_row = 0; mat_row < Order(local_B); mat_row++)
                   *(temp+mat_row) = Entry(local_B, mat_row, mat_col);       // switch rows and colums in local_B, for column major storage
            
            MPI_Send(temp, Order(local_B), FLOAT_MPI, 0, 0, grid->comm);
        }

        free(temp);
    }
                     
}  /* Print_matrix_B */


/*********************************************************/
/* Recive and Print Matrix A:  
 *     foreach global row of the matrix,
 *         foreach grid column 
 *             send n_bar floats to process 0 from each other process
 *             receive a block of n_bar floats on process 0 from other processes and print them 
 */
void Print_matrix_C(
         char*            title    /* in  */,  
         LOCAL_MATRIX_T*  local_C  /* out */,
         GRID_INFO_T*     grid     /* in  */,
         int              n        /* in  */) {

    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        source;
    int        coords[2];
    FLOAT*     temp;
    MPI_Status status;

    if (grid->my_rank == 0) {
        temp = (FLOAT*) malloc(Order(local_C)*sizeof(FLOAT));
        printf("%s\n", title);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Order(local_C);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &source);
                if (source == 0) {
                    for(mat_col = 0; mat_col < Order(local_C); mat_col++)
                        printf("%20.15E ", Entry(local_C, mat_row, mat_col));
                } else {
                    MPI_Recv(temp, Order(local_C), FLOAT_MPI, source, 0,
                        grid->comm, &status);
                    for(mat_col = 0; mat_col < Order(local_C); mat_col++)
                        printf("%20.15E ", temp[mat_col]);
                }
            }
            printf("\n");
        }
        free(temp);
    } else {
        for (mat_row = 0; mat_row < Order(local_C); mat_row++) 
            MPI_Send(&Entry(local_C, mat_row, 0), Order(local_C), 
                FLOAT_MPI, 0, 0, grid->comm);
    }
                     
}  /* Print_matrix_C */


/*********************************************************/
/* Recive and Write Matrix C into a file:  
 *     foreach global row of the matrix,
 *         foreach grid column 
 *             send n_bar floats to process 0 from each other process
 *             receive a block of n_bar floats on process 0 from other processes and print them 
 */
void Write_matrix_C(
         char*            title    /* in  */,  
         LOCAL_MATRIX_T*  local_C  /* out */,
         GRID_INFO_T*     grid     /* in  */,
         int              n        /* in  */) {
    
    FILE      *fp;
    int        mat_row, mat_col;
    int        grid_row, grid_col;
    int        source;
    int        coords[2];
    FLOAT*     temp;
    MPI_Status status;

    if (grid->my_rank == 0) {
        fp = fopen("C.dat", "w+");
        temp = (FLOAT*) malloc(Order(local_C)*sizeof(FLOAT));
        printf("%s\n", title);
        for (mat_row = 0;  mat_row < n; mat_row++) {
            grid_row = mat_row/Order(local_C);
            coords[0] = grid_row;
            for (grid_col = 0; grid_col < grid->q; grid_col++) {
                coords[1] = grid_col;
                MPI_Cart_rank(grid->comm, coords, &source);
                if (source == 0) {
                    for(mat_col = 0; mat_col < Order(local_C); mat_col++)
                        fprintf(fp, "%20.15E ", Entry(local_C, mat_row, mat_col));
                        // printf("%20.15E ", Entry(local_A, mat_row, mat_col));
                } else {
                    MPI_Recv(temp, Order(local_C), FLOAT_MPI, source, 0,
                        grid->comm, &status);
                    for(mat_col = 0; mat_col < Order(local_C); mat_col++)
                        fprintf(fp, "%20.15E ", temp[mat_col]);
                        // printf("%20.15E ", temp[mat_col]);
                }
            }
            fprintf(fp,"\n");
        }
        free(temp);
        fclose(fp);
    } else {
        for (mat_row = 0; mat_row < Order(local_C); mat_row++) 
            MPI_Send(&Entry(local_C, mat_row, 0), Order(local_C), 
                FLOAT_MPI, 0, 0, grid->comm);
    }
                     
}  /* Write_matrix_C */


/*********************************************************/
/*
*  Set local matrix's element to zero
*/
void Set_to_zero(
         LOCAL_MATRIX_T*  local_A  /* out */) {

    int i, j;

    for (i = 0; i < Order(local_A); i++)
        for (j = 0; j < Order(local_A); j++)
            Entry(local_A,i,j) = 0.0E0;

}  /* Set_to_zero */


/*********************************************************/
void Build_matrix_type(
         LOCAL_MATRIX_T*  local_A  /* in */) {
    MPI_Datatype  temp_mpi_t;
    int           block_lengths[2];
    MPI_Aint      displacements[2];
    MPI_Datatype  typelist[2];
    MPI_Aint      start_address;
    MPI_Aint      address;

    MPI_Type_contiguous(Order(local_A)*Order(local_A), 
        FLOAT_MPI, &temp_mpi_t);                         // Creates a contiguous datatype
    /*
    Synopsis

           int MPI_Type_contiguous(int count,
                             MPI_Datatype oldtype,
                             MPI_Datatype *newtype)

    Input Parameters

    count
           replication count (nonnegative integer)
    oldtype
           old datatype (handle)
    */
    block_lengths[0] = block_lengths[1] = 1;
   
    typelist[0] = MPI_INT;
    typelist[1] = temp_mpi_t;

    MPI_Address(local_A, &start_address);                 // Gets the address of a location in caller's memory
    MPI_Address(&(local_A->n_bar), &address);
    /*
    Synopsis

           int MPI_Address(const void *location, MPI_Aint *address)
    
    Input Parameters

    location
           location in caller memory (choice)
    
    Output Parameters

           address
           address of location (address integer)
    */
    displacements[0] = address - start_address;
    
    MPI_Address(local_A->entries, &address);
    displacements[1] = address - start_address;

    MPI_Type_struct(2, block_lengths, displacements,
        typelist, &local_matrix_mpi_t);                   // Creates a struct datatype
    /*
    Synopsis

    int MPI_Type_struct(int count,
                      const int *array_of_blocklengths,
                      const MPI_Aint *array_of_displacements,
                      const MPI_Datatype *array_of_types,
                      MPI_Datatype *newtype)
    
    Input Parameters

    count
        number of blocks (integer) -- also number of entries in arrays array_of_types , array_of_displacements and array_of_blocklengths
    array_of_blocklengths
        number of elements in each block (array)
    array_of_displacements
        byte displacement of each block (array)
    array_of_types
        type of elements in each block (array of handles to datatype objects)
    
    Output Parameters
    
    newtype
        new datatype (handle)
    */
    MPI_Type_commit(&local_matrix_mpi_t);                 // Commits the datatype
    /*
    Synopsis

    int MPI_Type_commit(MPI_Datatype *datatype)
    
    Input Parameters

    datatype
        datatype (handle)
    */
}  /* Build_matrix_type */


/*********************************************************/
/* local matrix multiplication function 
*  withing OpenMP Thread Acceleration
*/
void Local_matrix_multiply(
         LOCAL_MATRIX_T*  local_A  /* in  */,
         LOCAL_MATRIX_T*  local_B  /* in  */, 
         LOCAL_MATRIX_T*  local_C  /* out */) {
    
    int i, j, k;
    
    // int my_rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);             // Get my process id in the MPI communicator
    
    #pragma omp parallel for private(i, j, k) shared(local_A, local_B, local_C) num_threads(NUM_THREADS)       // Threads acceleration upgrade, parallel task split
    for (i = 0; i < Order(local_A); i++) {
        // printf("Current in the Fox Kernel:\n my process id is %d, my thread id is %d\n",my_rank,omp_get_thread_num());                                      
        for (j = 0; j < Order(local_A); j++)              
            for (k = 0; k < Order(local_B); k++)
                Entry(local_C,i,j) = Entry(local_C,i,j)             // switch rows and colums in local_B, for column major storage
                    + Entry(local_A,i,k)*Entry(local_B,j,k);        // continuous memory access, local matrix multiplication A(i,k)*B^T(j,k)                                                                         
               /* Entry(local_C,i,j) = Entry(local_C,i,j) 
                    + Entry(local_A,i,k)*Entry(local_B,k,j);        // non-continuous memory access, A(i,k)*B^T(j,k) is more proper
               */
    }
}  /* Local_matrix_multiply */


/*********************************************************/
/* Recive and Print Local Matrix A:  
 *        Process 0 print local matrix local_A    
 *        Other Processess send local matrix local_A to process 0
 *        And process 0 receive local matrix local_A from other processess      
 */
void Print_local_matrices_A(
         char*            title    /* in */,
         LOCAL_MATRIX_T*  local_A  /* in */, 
         GRID_INFO_T*     grid     /* in */) {

    int         coords[2];
    int         i, j;
    int         source;
    MPI_Status  status;
    
    // print by process No.0 in process mesh
    if (grid->my_rank == 0) {
        printf("%s\n", title);
        printf("Process %d > grid_row = %d, grid_col = %d\n",
            grid->my_rank, grid->my_row, grid->my_col);
        for (i = 0; i < Order(local_A); i++) {
            for (j = 0; j < Order(local_A); j++)
                printf("%20.15E ", Entry(local_A,i,j));
            printf("\n");
        }
        for (source = 1; source < grid->p; source++) {
            MPI_Recv(temp_mat, 1, local_matrix_mpi_t, source, 0,
                grid->comm, &status);
            MPI_Cart_coords(grid->comm, source, 2, coords);
            printf("Process %d > grid_row = %d, grid_col = %d\n",
                source, coords[0], coords[1]);
            for (i = 0; i < Order(temp_mat); i++) {
                for (j = 0; j < Order(temp_mat); j++)
                    printf("%20.15E ", Entry(temp_mat,i,j));
                printf("\n");
            }
        }
        fflush(stdout);
    } else {
        MPI_Send(local_A, 1, local_matrix_mpi_t, 0, 0, grid->comm);
    }
        
}  /* Print_local_matrices_A */



/*********************************************************/
/* Recive and Print Local Matrix for local matrix B's transpose:
 *        Process 0 print local matrix local_A    
 *        Other Processess send local matrix local_A to process 0
 *        And process 0 receive local matrix local_A from other processess      
 */
void Print_local_matrices_B(
         char*            title    /* in */,
         LOCAL_MATRIX_T*  local_B  /* in */, 
         GRID_INFO_T*     grid     /* in */) {

    int         coords[2];
    int         i, j;
    int         source;
    MPI_Status  status;
    
    // print by process No.0 in process mesh
    if (grid->my_rank == 0) {
        printf("%s\n", title);
        printf("Process %d > grid_row = %d, grid_col = %d\n",
            grid->my_rank, grid->my_row, grid->my_col);
        for (i = 0; i < Order(local_B); i++) {
            for (j = 0; j < Order(local_B); j++)
                printf("%20.15E ", Entry(local_B,j,i));                   // switch rows and colums in local_B, for column major storage
            printf("\n");
        }
        for (source = 1; source < grid->p; source++) {
            MPI_Recv(temp_mat, 1, local_matrix_mpi_t, source, 0,
                grid->comm, &status);
            MPI_Cart_coords(grid->comm, source, 2, coords);
            printf("Process %d > grid_row = %d, grid_col = %d\n",
                source, coords[0], coords[1]);
            for (i = 0; i < Order(temp_mat); i++) {
                for (j = 0; j < Order(temp_mat); j++)
                    printf("%20.15E ", Entry(temp_mat,j,i));             // switch rows and colums in local_B, for column major storage
                printf("\n");
            }
        }
        fflush(stdout);
    } else {
        MPI_Send(local_B, 1, local_matrix_mpi_t, 0, 0, grid->comm);
    }
        
}  /* Print_local_matrices_B */


/*********************************************************/
/* Recive and Print Local Matrix A:  
 *        Process 0 print local matrix local_A    
 *        Other Processess send local matrix local_A to process 0
 *        And process 0 receive local matrix local_A from other processess      
 */
void Print_local_matrices_C(
         char*            title    /* in */,
         LOCAL_MATRIX_T*  local_C  /* in */, 
         GRID_INFO_T*     grid     /* in */) {

    int         coords[2];
    int         i, j;
    int         source;
    MPI_Status  status;
    
    // print by process No.0 in process mesh
    if (grid->my_rank == 0) {
        printf("%s\n", title);
        printf("Process %d > grid_row = %d, grid_col = %d\n",
            grid->my_rank, grid->my_row, grid->my_col);
        for (i = 0; i < Order(local_C); i++) {
            for (j = 0; j < Order(local_C); j++)
                printf("%20.15E ", Entry(local_C,i,j));
            printf("\n");
        }
        for (source = 1; source < grid->p; source++) {
            MPI_Recv(temp_mat, 1, local_matrix_mpi_t, source, 0,
                grid->comm, &status);
            MPI_Cart_coords(grid->comm, source, 2, coords);
            printf("Process %d > grid_row = %d, grid_col = %d\n",
                source, coords[0], coords[1]);
            for (i = 0; i < Order(temp_mat); i++) {
                for (j = 0; j < Order(temp_mat); j++)
                    printf("%20.15E ", Entry(temp_mat,i,j));
                printf("\n");
            }
        }
        fflush(stdout);
    } else {
        MPI_Send(local_C, 1, local_matrix_mpi_t, 0, 0, grid->comm);
    }
        
}  /* Print_local_matrices_C */


/*********************************************************/
/* Recive and Write Local Matrix A:  
 *        Process 0 print local matrix local_A    
 *        Other Processess send local matrix local_A to process 0
 *        And process 0 receive local matrix local_A from other processess      
 */
void Write_local_matrices_A(
         char*            title    /* in */,
         LOCAL_MATRIX_T*  local_A  /* in */, 
         GRID_INFO_T*     grid     /* in */) {
    
    FILE        *fp;
    int         coords[2];
    int         i, j;
    int         source;
    MPI_Status  status;
    
    // print by process No.0 in process mesh
    if (grid->my_rank == 0) {
        fp = fopen("local_A.dat","w+");
        printf("%s\n", title);
        
        fprintf(fp,"Process %d > grid_row = %d, grid_col = %d\n",
            grid->my_rank, grid->my_row, grid->my_col);
        for (i = 0; i < Order(local_A); i++) {
            for (j = 0; j < Order(local_A); j++)
                fprintf(fp,"%20.15E ", Entry(local_A,i,j));
            fprintf(fp, "\n");
        }
        for (source = 1; source < grid->p; source++) {
            MPI_Recv(temp_mat, 1, local_matrix_mpi_t, source, 0,
                grid->comm, &status);
            MPI_Cart_coords(grid->comm, source, 2, coords);
            fprintf(fp, "Process %d > grid_row = %d, grid_col = %d\n",
                source, coords[0], coords[1]);
            for (i = 0; i < Order(temp_mat); i++) {
                for (j = 0; j < Order(temp_mat); j++)
                    fprintf(fp, "%20.15E ", Entry(temp_mat,i,j));
                fprintf(fp, "\n");
            }
        }
        fflush(stdout);
        fclose(fp);
    } else {
        MPI_Send(local_A, 1, local_matrix_mpi_t, 0, 0, grid->comm);
    }
        
}  /* Write_local_matrices_A */



/*********************************************************/
/* Recive and Write Local Matrix for local matrix B's transpose:
 *        Process 0 print local matrix local_A    
 *        Other Processess send local matrix local_A to process 0
 *        And process 0 receive local matrix local_A from other processess      
 */
void Write_local_matrices_B(
         char*            title    /* in */,
         LOCAL_MATRIX_T*  local_B  /* in */, 
         GRID_INFO_T*     grid     /* in */) {
    
    FILE        *fp;
    int         coords[2];
    int         i, j;
    int         source;
    MPI_Status  status;
    
    // print by process No.0 in process mesh
    if (grid->my_rank == 0) {
        fp = fopen("local_B.dat","w+");
        printf("%s\n", title);

        fprintf(fp, "Process %d > grid_row = %d, grid_col = %d\n",
            grid->my_rank, grid->my_row, grid->my_col);
        for (i = 0; i < Order(local_B); i++) {
            for (j = 0; j < Order(local_B); j++)
                fprintf(fp, "%20.15E ", Entry(local_B,j,i));                   // switch rows and colums in local_B, for column major storage
            fprintf(fp, "\n");
        }
        for (source = 1; source < grid->p; source++) {
            MPI_Recv(temp_mat, 1, local_matrix_mpi_t, source, 0,
                grid->comm, &status);
            MPI_Cart_coords(grid->comm, source, 2, coords);
            fprintf(fp, "Process %d > grid_row = %d, grid_col = %d\n",
                source, coords[0], coords[1]);
            for (i = 0; i < Order(temp_mat); i++) {
                for (j = 0; j < Order(temp_mat); j++)
                    fprintf(fp, "%20.15E ", Entry(temp_mat,j,i));             // switch rows and colums in local_B, for column major storage
                fprintf(fp, "\n");
            }
        }
        fflush(stdout);
        fclose(fp);
    } else {
        MPI_Send(local_B, 1, local_matrix_mpi_t, 0, 0, grid->comm);
    }
        
}  /* Write_local_matrices_B */


/*********************************************************/
/* Recive and Write Local Matrix C:  
 *        Process 0 print local matrix local_C    
 *        Other Processess send local matrix local_C to process 0
 *        And process 0 receive local matrix local_C from other processess      
 */
void Write_local_matrices_C(
         char*            title    /* in */,
         LOCAL_MATRIX_T*  local_C  /* in */, 
         GRID_INFO_T*     grid     /* in */) {
    
    FILE        *fp;
    int         coords[2];
    int         i, j;
    int         source;
    MPI_Status  status;
    
    // print by process No.0 in process mesh
    if (grid->my_rank == 0) {
        fp = fopen("local_C.dat","w+");
        printf("%s\n", title);

        fprintf(fp, "Process %d > grid_row = %d, grid_col = %d\n",
            grid->my_rank, grid->my_row, grid->my_col);
        for (i = 0; i < Order(local_C); i++) {
            for (j = 0; j < Order(local_C); j++)
                fprintf(fp, "%20.15E ", Entry(local_C,i,j));
            fprintf(fp, "\n");
        }
        for (source = 1; source < grid->p; source++) {
            MPI_Recv(temp_mat, 1, local_matrix_mpi_t, source, 0,
                grid->comm, &status);
            MPI_Cart_coords(grid->comm, source, 2, coords);
            fprintf(fp, "Process %d > grid_row = %d, grid_col = %d\n",
                source, coords[0], coords[1]);
            for (i = 0; i < Order(temp_mat); i++) {
                for (j = 0; j < Order(temp_mat); j++)
                    fprintf(fp, "%20.15E ", Entry(temp_mat,i,j));
                fprintf(fp, "\n");
            }
        }
        fflush(stdout);
        fclose(fp);
    } else {
        MPI_Send(local_C, 1, local_matrix_mpi_t, 0, 0, grid->comm);
    }
        
}  /* Write_local_matrices_C */
