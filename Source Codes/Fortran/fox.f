C  fox.f -- uses Fox's algorithm to multiply two square matrices
C  FORTRAN 77 Fixed format Code
C  
C  Input:
C      n: global order of matrices
C      A,B: the factor matrices (in row-major order)
C  Output:
C      C: the product matrix
C 
C  Notes:
C      1.  Assumes the number of processes is a perfect square
C      2.  Assumes that n is evenly divisible by sqrt(p)
C 
C  See Chap 7, pp. 113 & ff and pp. 125 & ff in PPMPI
C 
      PROGRAM ParFox
      INCLUDE 'mpif.h'
      integer              ierr
C change the size of MAX to allow more entries.  It is
C limited to 1000 here for purposes of memory usage while
C testing
      integer              MAX
      parameter            ( MAX=1000 )
C grid array representation of each subscript
      integer              P, COMM, ROW_COMM, COL_COMM, Q
      integer              MY_ROW, MY_COL, MYRANK
      parameter            (P=1, COMM = 2, ROW_COMM = 3,
     +                      COL_COMM = 4, Q = 5, 
     +                      MY_ROW=6, MY_COL=7, MYRANK=8)      
      integer     	   grid_info(8)
      integer              n
      integer              n_bar
      integer              my_rank
      real                 A(MAX*MAX)
      real                 B(MAX*MAX)
      real                 C(MAX*MAX)
C
      call MPI_INIT( ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD,  my_rank, ierr )
C
      call Setup_grid( grid_info)
      if (my_rank .EQ. 0)  then
          print *, 'What is the order of the matrices?'
          read *, n
      endif
C
      call MPI_BCAST(n,1,MPI_INTEGER, 0, MPI_COMM_WORLD,ierr)
      n_bar = n/grid_info(Q)
C
      call Read_matrix('Enter A ', A,grid_info, n , n_bar)
      call Print_matrix('We read A =', A,grid_info, 
     +                          n, n_bar)
C       
      call Read_matrix('Enter B ', B,grid_info,n , n_bar)
      call Print_matrix('We read B =', B,grid_info,
     +                          n, n_bar )
C 
      call Fox(n,  grid_info, A, B, C, n_bar)
C
      call Print_matrix('The product is', C, 
     +                           grid_info, n, n_bar)
C
      call MPI_FINALIZE(ierr)
      end
C
C
C *******************************************************  
      subroutine Setup_grid(grid)
C grid array representation of each subscript
      integer              P, COMM, ROW_COMM, COL_COMM, Q
      integer              MY_ROW, MY_COL, MYRANK
      parameter            (P=1, COMM = 2, ROW_COMM = 3,
     +                      COL_COMM = 4, Q= 5, 
     +                      MY_ROW=6, MY_COL=7, MYRANK=8)      
      integer   grid(8)
      INCLUDE 'mpif.h'
      integer ierr
      integer old_rank
      integer dimensions(0:1)
      logical wrap_around(0:1)
      integer coordinates(0:1)
      integer free_coords(0:1)
      real    gridreal
C
C  Set up Global Grid Information   
      call MPI_COMM_SIZE(MPI_COMM_WORLD,  grid(P), ierr )
      call MPI_COMM_RANK(MPI_COMM_WORLD,  old_rank, ierr )
C
C  We assume p is a perfect square
      gridreal = grid(P)   
      grid(Q) =  sqrt( gridreal )
      dimensions(0) = grid(Q)
      dimensions(1) = grid(Q)
C
C  We want a circular shift in second dimension.   
C  Don't care about first                          
      wrap_around(0) = .true.
      wrap_around(1) = .true.
      call MPI_CART_CREATE(MPI_COMM_WORLD, 2, dimensions,
     +     wrap_around, 1,  grid(COMM), ierr)
      call MPI_COMM_RANK(grid(COMM),  grid(MYRANK), ierr)
      call MPI_CART_COORDS(grid(COMM), grid(MYRANK), 2,
     +                  coordinates, ierr)
      grid(MY_ROW) = coordinates(0)
      grid(MY_COL) = coordinates(1)
C
C  Set up row communicators   
      free_coords(0) = 0
      free_coords(1) = 1
      call MPI_CART_SUB(grid(COMM), free_coords,
     +          grid(ROW_COMM), ierr)
C
C  Set up column communicators   
      free_coords(0) = 1
      free_coords(1) = 0
      call MPI_CART_SUB(grid(COMM), free_coords,
     +      grid(COL_COMM), ierr)
      return
      end 
C
C
C *******************************************************  
      subroutine Fox(n, grid, local_A, local_B, local_C, 
     +                      Order)
      integer              MAX
      parameter            ( MAX=1000 )  
      integer              n
      integer              Order       
      real  		   local_A(Order*Order)    
      real  		   local_B(Order*Order)    
      real  		   local_C(Order*Order)
C grid array representation of each subscript
      integer              P, COMM, ROW_COMM, COL_COMM, Q
      integer              MY_ROW, MY_COL, MYRANK
      parameter            (P=1, COMM = 2, ROW_COMM = 3,
     +                      COL_COMM = 4, Q = 5, 
     +                      MY_ROW=6, MY_COL=7, MYRANK=8)        
      integer     	   grid(8)
      INCLUDE 'mpif.h'
      integer ierr
      real                 temp_A(MAX*MAX)   
C  matrix of A used during    
C  the current stage         
      integer              stage
      integer              bcast_root
      integer              n_bar    
      integer              source
      integer              dest
      integer              status(MPI_STATUS_SIZE)
C
      n_bar = Order
      call Set_to_zero(local_C, n_bar)
C
C  Calculate addresses for circular shift of B     
      source = MOD(grid(MY_ROW) + 1, grid(Q)  )
      dest =   MOD(grid(MY_ROW) + grid(Q) - 1, grid(Q) )
C
      do 100 stage = 0, grid(Q)-1
          bcast_root = MOD( grid(MY_ROW) + stage, grid(Q)  )
          if (bcast_root .EQ. grid(MY_COL))  then
              call MPI_BCAST(local_A,n_bar*n_bar,MPI_REAL,
     +             bcast_root, grid(ROW_COMM), ierr)
              call Local_matrix_multiply(local_A, local_B,
     +             local_C, n_bar)
          else  
              call MPI_BCAST(temp_A,n_bar*n_bar,MPI_REAL,
     +             bcast_root, grid(ROW_COMM), ierr)
              call Local_matrix_multiply(temp_A, local_B,
     +             local_C, n_bar)
          endif
          call MPI_SENDRECV_REPLACE(local_B, n_bar*n_bar, 
     +         MPI_REAL,
     +         dest, 0, source, 0, grid(COL_COMM), status, ierr)
 100  continue  
C
      return
      end      
C
C *******************************************************  
C  Read and distribute matrix:  
C      foreach global row of the matrix,
C          foreach grid column 
C              read a block of n_bar floats on process 0
C              and send them to the appropriate process.
C 
      subroutine Read_matrix(prompt, local_A, grid, n, Order)
      integer                  MAX
      parameter               ( MAX=1000 )
      character *10            prompt    
      integer                  Order
      real                     local_A(Order* Order)  
      integer                  n         
C grid array representation of each subscript
      integer              P, COMM, ROW_COMM, COL_COMM, Q
      integer              MY_ROW, MY_COL, MYRANK
      parameter            (P=1, COMM = 2, ROW_COMM = 3,
     +                      COL_COMM = 4, Q = 5, 
     +                      MY_ROW=6, MY_COL=7, MYRANK=8)       
      integer              grid(8)
      INCLUDE        'mpif.h'
      integer        ierr
      integer        mat_row, mat_col
      integer        grid_row, grid_col
      integer        dest
      integer        coords(0:1)
      real           temp( MAX )
C
      integer   status(MPI_STATUS_SIZE)
      data   dest/ 0/
       
      if (grid(MYRANK) .EQ. 0)  then
          print *, prompt, '(One element per line)'
          do 100 mat_row = 1, n   
              grid_row = (mat_row-1)/Order 
              coords(0) = grid_row
              do 200 grid_col = 1, grid(Q)   
                  coords(1) = grid_col - 1
                  call MPI_CART_RANK(grid(COMM), 
     +                              coords,dest,ierr)
                  if (dest .EQ. 0)  then
C                   do 300 mat_col =1, Order  
                    read *, 
     +              (local_A((mat_row-1)*Order+mat_col),
     +               mat_col = 1, Order)
C 300                continue          
                  else  
C                      do 400 mat_col = 1, Order  
                       read *,(temp(mat_col), mat_col =1,Order)  
C 400                  continue 
                      call MPI_SEND(temp,Order,MPI_REAL, 
     +                      dest, 0,grid(COMM), ierr)
                  endif
 200          continue
 100      continue     
      else  
          do 500 mat_row = 1, Order
               call MPI_RECV(local_A(1+(mat_row-1)*Order),
     +              Order, MPI_REAL, 
     +              0, 0, grid(COMM),  status, ierr)
 500      continue
      endif
C
      return
      end   
C
C
C *******************************************************  
      subroutine Print_matrix(title, local_A, grid, n, Order)
      integer              MAX
      parameter            ( MAX=1000 )
      character *14 title   
      integer       Order  
      real          local_A(Order *Order)     
      integer       n
C grid array representation of each subscript
      integer              P, COMM, ROW_COMM, COL_COMM, Q
      integer              MY_ROW, MY_COL, MYRANK
      parameter            (P=1, COMM = 2, ROW_COMM = 3,
     +                      COL_COMM = 4, Q = 5, 
     +                      MY_ROW=6, MY_COL=7, MYRANK=8)      
      integer       grid(8) 
      INCLUDE       'mpif.h'
      integer        ierr      
      integer        mat_row, mat_col
      integer        grid_row, grid_col
      integer        source
      integer        coords(0:1)
      real           temp(MAX)
      real           printrow(MAX)
      integer        cnt
      integer        status(MPI_STATUS_SIZE)
C
      if (grid(MYRANK) .EQ. 0)  then
          print *, title
          do 100 mat_row = 1, n  
              cnt = 0
              grid_row = (mat_row-1)/Order 
              coords(0) = grid_row
              do 200 grid_col =1, grid(Q)  
                 coords(1) = grid_col-1
                 call MPI_CART_RANK(grid(COMM),coords,
     +                        source, ierr)
                 if (source .EQ. 0)  then
                    do 300 mat_col = 1, Order 
                        printrow(cnt+mat_col) = 
     +                     local_A((mat_row-1)*Order+mat_col)
 300                continue          
                    cnt = cnt + Order
                 else  
                     call MPI_RECV(temp, Order, MPI_REAL, 
     +                     source, 0,grid(COMM),  status, ierr) 
                     do 350  mat_col = 1, Order
                          printrow(cnt+mat_col)= temp(mat_col) 
 350                 continue
                     cnt = cnt + Order
                 endif
 200          continue
              print 400, (printrow(mat_col), mat_col = 1,n)
 400          format (' ',1000F10.2)
 100      continue
      else
          do 410 mat_row = 1,Order
              call MPI_SEND(local_A(1+ (mat_row-1)*Order ),
     +                  Order, MPI_REAL , 0, 0, grid(COMM), ierr)
 410      continue 
      endif
C
      return
      end   
C
C
C *******************************************************  
      subroutine Set_to_zero(local_A, Order)
      integer              Order
      real                 local_A(Order*Order)   

C
      integer i, j
C
      do 100 i = 1, Order 
          do 200 j = 1,Order 
               local_A((i-1)*j) = 0.0
 200      continue
 100  continue
C
      return
      end  
C *******************************************************  
      subroutine Local_matrix_multiply(local_A, 
     +                   local_B, local_C, Order)
      integer           Order
      real  local_A(Order* Order)   
      real  local_B(Order* Order)   
      real  local_C(Order* Order)
C
      integer i, j, k
C
      do 100 i = 1, Order 
          do 200 j =1,  Order 
              do 300 k = 1, Order 
                local_C( (i-1)* Order + j) = 
     +                  local_C((i-1)*Order +j) + 
     +                  local_A((i-1)*Order + k) *
     +                  local_B((k-1)*Order + j)
 300          continue
 200      continue
 100  continue
C
      return
      end   
C
