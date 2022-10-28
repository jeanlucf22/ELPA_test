#include <elpa/elpa.h>

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define SINGLETYPE
//#define DOUBLETYPE
//#define SCOMPLEXTYPE
//#define DCOMPLEXTYPE

#ifdef DOUBLETYPE
typedef double mattype;
typedef double evtype;
#endif
#ifdef SINGLETYPE
typedef float mattype;
typedef float evtype;
#endif
#ifdef SCOMPLEXTYPE
typedef float complex mattype;
typedef float evtype;
#endif
#ifdef DCOMPLEXTYPE
typedef double complex mattype;
typedef double evtype;
#endif

int main(int argc, char*argv[])
{
   /* mpi */
   int nprocs, myid;
   int my_prow, my_pcol;

   int error_elpa;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    assert(argc == 2);

    /* matrix dimensions */
    int na = atoi(argv[1]);

    /* number of eigenvectors that should be computed ( 1<= nev <= na) */
    int nev = na;


    if (myid == 0)printf("==========================================\n");
    if (myid == 0)printf("ELPA solve\n");

    int np_cols = (int)sqrt(nprocs);
    int np_rows = nprocs/np_cols;
    if(np_cols*np_rows != nprocs)
    {
        printf("Number of MPI tasks should be a square number\n");
        exit(1);
    }

    char order = 'C';
    int my_blacs_ctxt = Csys2blacs_handle(MPI_COMM_WORLD);
    Cblacs_gridinit(&my_blacs_ctxt, &order, np_rows, np_cols);
    Cblacs_gridinfo(my_blacs_ctxt, &np_rows, &np_cols, &my_prow, &my_pcol);

    int na_rows = na/np_rows;
    int na_cols = na/np_cols;
    if(na_rows*np_rows != na)
    {
        printf("Number of MPI tasks/row should divide matrix size\n");
        exit(1);
    }
    // size of the BLACS block cyclic distribution (needs to be a power of 2)
    int nblk = 16;

    if (myid == 0) {
    printf("Matrix size: %d\n", na);
    printf("Number of MPI process rows: %d\n", np_rows);
    printf("Number of MPI process cols: %d\n", np_cols);
    }

   /* allocate the matrices needed for elpa */
   mattype* a  = calloc(na_rows*na_cols, sizeof(mattype));
   mattype* z  = calloc(na_rows*na_cols, sizeof(mattype));
   evtype* ev = calloc(na, sizeof(evtype));

    if(my_prow==my_pcol)
    for (int i = 0; i < na_rows; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            a[i * na_cols + j] = rand() / (mattype) RAND_MAX;
            a[j * na_cols + i] = a[i * na_cols + j];
        }
    }
    else
    for (int i = 0; i < na_rows; i++)
    {
        for (int j = 0; j < na_cols; j++)
        {
            if(my_prow>my_pcol)a[i * na_cols + j] = rand() / (mattype) RAND_MAX;
            else               a[j * na_cols + i] = rand() / (mattype) RAND_MAX;
        }
    }


/* check the error code of all ELPA functions */
   if (elpa_init(ELPA_API_VERSION) != ELPA_OK) {
     fprintf(stderr, "Error: ELPA API version not supported");
     exit(1);
   }

   elpa_t handle = elpa_allocate(&error_elpa);
   /* Set parameters */
   elpa_set(handle, "na", (int) na, &error_elpa);
   assert(error_elpa==ELPA_OK);

   elpa_set(handle, "nev", (int) nev, &error_elpa);
   assert(error_elpa==ELPA_OK);

   if (myid == 0) {
     printf("Setting the matrix parameters na=%d, nev=%d \n",na,nev);
   }
   elpa_set(handle, "local_nrows", (int) na_rows, &error_elpa);
   assert(error_elpa==ELPA_OK);

   elpa_set(handle, "local_ncols", (int) na_cols, &error_elpa);
   assert(error_elpa==ELPA_OK);

   elpa_set(handle, "nblk", (int) nblk, &error_elpa);
   assert(error_elpa==ELPA_OK);

   elpa_set(handle, "mpi_comm_parent", (int) (MPI_Comm_c2f(MPI_COMM_WORLD)), &error_elpa);
   assert(error_elpa==ELPA_OK);

   elpa_set(handle, "process_row", (int) my_prow, &error_elpa);
   assert(error_elpa==ELPA_OK);

   elpa_set(handle, "process_col", (int) my_pcol, &error_elpa);
   assert(error_elpa==ELPA_OK);

   MPI_Barrier(MPI_COMM_WORLD);
 
   /* Setup */
   if (myid == 0) {
     printf("Setup ELPA\n");
   }
   int success = elpa_setup(handle);
    assert(success==ELPA_OK);

    elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &error_elpa);
    assert(error_elpa==ELPA_OK);

    elpa_set(handle, "gpu", 1, &error_elpa);
    assert(error_elpa==ELPA_OK);

    int value;
    elpa_get(handle, "solver", &value, &error_elpa);
    if (myid == 0) {
        printf("Solver is set to %d \n", value);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double starttime = MPI_Wtime();

    /* Solve EV problem */
#ifdef SINGLETYPE
    elpa_eigenvectors_float(handle, a, ev, z, &error_elpa);
#endif
#ifdef DOUBLETYPE
    elpa_eigenvectors_double(handle, a, ev, z, &error_elpa);
#endif
#ifdef SCOMPLEXTYPE
    elpa_eigenvectors_float_complex(handle, a, ev, z, &error_elpa);
#endif
#ifdef DCOMPLEXTYPE
    elpa_eigenvectors_double_complex(handle, a, ev, z, &error_elpa);
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    double endtime   = MPI_Wtime();

    assert(error_elpa==ELPA_OK);

    if (myid == 0) {
        printf("Solver took %f seconds\n",endtime-starttime);
    }

    if (myid == 0) {
        printf("Eigenvalues\n");
        for(int i=0;i<10;i++)printf("eigenvalue %d = %le\n", i, ev[i]);
    }

    if (myid == 0) {
        printf("Deallocate\n");
    }

    elpa_deallocate(handle, &error_elpa);
    elpa_uninit(&error_elpa);

    free(a);
    free(z);
    free(ev);

    if (myid == 0) {
        printf("Finalize\n");
    }

    MPI_Finalize();

    return 0;
}
