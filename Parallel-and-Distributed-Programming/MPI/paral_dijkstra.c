/* File:   paral_dijkstra.c derived from mpi_io.c
 * Purpose:  Implement I/O functions that will be useful in an
 *           an MPI implementation of Dijkstra's algorithm.  
 *           In particular, the program creates an MPI_Datatype
 *           that can be used to implement input and output of
 *           a matrix that is distributed by block columns.  It
 *           also implements input and output functions that use
 *           this datatype.  Finally, it implements a function
 *           that prints out a process' submatrix as a string.
 *           This makes it more likely that printing the submatrix 
 *           assigned to one process will be printed without 
 *           interruption by another process.
 *
 * Compile:  mpicc -g -Wall -o mpi_io mpi_io.c
 * Run:      mpiexec -n <p> ./mpi_io (on lab machines)
 *           csmpiexec -n <p> ./mpi_io (on the penguin cluster)
 *
 * Input:    n:  the number of rows and the number of columns 
 *               in the matrix
 *           mat:  the matrix:  note that INFINITY should be
 *               input as 1000000
 * Output:   The submatrix assigned to each process and the
 *           complete matrix printed from process 0.  Both
 *           print "i" instead of 1000000 for infinity.
 *
 * Notes:
 * 1.  The number of processes, p, should evenly divide n.
 * 2.  You should free the MPI_Datatype object created by
 *     the program with a call to MPI_Type_free:  see the
 *     main function.
 * 3.  Example:  Suppose the matrix is
 *
 *        0 1 2 3
 *        4 0 5 6 
 *        7 8 0 9
 *        8 7 6 0
 *
 *     Then if there are two processes, the matrix will be
 *     distributed as follows:
 *
 *        Proc 0:  0 1    Proc 1:  2 3
 *                 4 0             5 6
 *                 7 8             0 9
 *                 8 7             6 0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mpi.h>
#define MAX_STRING 10000
#define INFINITY 1000000

int Read_n(char *path, int my_rank, MPI_Comm comm);
MPI_Datatype Build_blk_col_type(int n, int loc_n);
void Read_matrix(char * path, int loc_mat[], int n, int loc_n, 
MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm, double *start_time);
void Print_dists(int dist[], int n);
void Print_paths(int pred[], int n);
int Find_min_loc_dist(int loc_dist[], int loc_known[], int loc_n);
void Print_local_matrix(int loc_mat[], int n, int loc_n, int my_rank);
void Print_matrix(int loc_mat[], int n, int loc_n,
      MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm);

int main(int argc, char* argv[]) {
	int *loc_mat, *loc_known, *loc_dist, *loc_pred;
	int n, loc_n, p, my_rank;
	MPI_Comm comm;
	MPI_Datatype blk_col_mpi_t;
	if (argc != 2) {
		printf("The number of arguments is wrong. Your should enter the test file name.\n");
		return -1;
	}


#  ifdef DEBUG
	int i, j;
#  endif
	double start_time, end_time;

	MPI_Init(&argc, &argv);
	comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &my_rank);
	
	n = Read_n(argv[1], my_rank, comm);
	if (n == -1) {
		return -1;
	}
	loc_n = n/p;
	loc_mat = malloc(n*loc_n*sizeof(int));

#  ifdef DEBUG
	printf("Proc %d > p = %d, n = %d, loc_n = %d\n",
			my_rank, p, n, loc_n);

	/* This ensures that the matrix elements are initialized when */
	/* debugging.  It shouldn't be necessary */
	for (i = 0; i < n; i++)
		for (j = 0; j < loc_n; j++)
			loc_mat[i*loc_n + j] = -1;
#  endif   
	
	/* Build the special MPI_Datatype before doing matrix I/O */
	blk_col_mpi_t = Build_blk_col_type(n, loc_n);
	Read_matrix(argv[1], loc_mat, n, loc_n, blk_col_mpi_t, my_rank, comm, &start_time);		

	start_time = clock();

	// Initialization for each process
	loc_known = malloc(loc_n * sizeof(int));
	loc_dist = malloc(loc_n * sizeof(int));
	loc_pred = malloc(loc_n * sizeof(int));
	int loc_v;
	for (loc_v = 0; loc_v < loc_n; loc_v++) {
		loc_dist[loc_v] = loc_mat[0 * loc_n + loc_v];
		loc_pred[loc_v] = 0;
		loc_known[loc_v] = 0;
	}
	if (my_rank == 0) {
		loc_known[0] = 1;
	}

	// dijkstra algorithm for each process
	int loc_iter, loc_u, my_min[2], glbl_min[2], new_dist;
	for (loc_iter = 1; loc_iter < n; loc_iter++) {
		loc_u = Find_min_loc_dist(loc_dist, loc_known, loc_n);
		if (loc_u == -1) {
			my_min[0] = INFINITY;
			my_min[1] = -1;		
		} else {
			my_min[0] = loc_dist[loc_u];
			my_min[1] = loc_u + my_rank * loc_n;					
		}
		MPI_Allreduce(my_min, glbl_min, 1, MPI_2INT, MPI_MINLOC, comm);
		if (my_rank == glbl_min[1] / loc_n) {
			loc_known[glbl_min[1] % loc_n] = 1;		
		}

		for (loc_v = 0; loc_v < loc_n; loc_v++) {
			if (!loc_known[loc_v]) {
				new_dist = glbl_min[0] + loc_mat[ glbl_min[1] * loc_n + loc_v ];
				if (new_dist < loc_dist[loc_v]) {
					loc_dist[loc_v] = new_dist;
					loc_pred[loc_v] = glbl_min[1];
				}
			}
		}

	}
	// gather all the information into process 0
	int * dist = NULL, * pred = NULL;
	if (my_rank == 0) {
		dist = malloc(n * sizeof(int));
		pred = malloc(n * sizeof(int));
	}

	end_time = (double)(clock());
	MPI_Gather(loc_dist, loc_n, MPI_INT, dist, loc_n, MPI_INT, 0, comm);
	MPI_Gather(loc_pred, loc_n, MPI_INT, pred, loc_n, MPI_INT, 0, comm);
	
	//end_time = (double)(clock());
	double elapsed_time = end_time - start_time;
	double total_time;
	MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
	if (my_rank == 0) {
		double real_time = (total_time / p) / CLOCKS_PER_SEC;
		printf("The elapsed time is: %lf.\n", real_time);
		Print_dists(dist, n);
		Print_paths(pred, n);
	}

	free(loc_mat);
	free(loc_known);
	free(loc_dist);
	free(loc_pred);
	free(dist);
	free(pred);
	
	/* When you're done with the MPI_Datatype, free it */
	MPI_Type_free(&blk_col_mpi_t);

	MPI_Finalize();
	return 0;
}  

int Find_min_loc_dist(int loc_dist[], int loc_known[], int loc_n) {
	int loc_u = -1, loc_v = 0;
	int min_dist = INFINITY;
	for (loc_v = 0; loc_v < loc_n; loc_v++) {
		if (!loc_known[loc_v]) {
			if (loc_dist[loc_v] < min_dist) {
				loc_u = loc_v;
				min_dist = loc_dist[loc_v];
			}
		}
	}
	return loc_u;
}

int Read_n(char * path, int my_rank, MPI_Comm comm) {
	int n;
	if (my_rank == 0){
		FILE * file;
		file = fopen(path, "r");
		if (file != NULL) {
			printf("The file is sucessfully readed.\n");
			fscanf(file, "%d", &n);
			fclose(file);
		} else {
			printf("The file doesn't exists or access is denied.\n");
			return -1;
		}
	}
	MPI_Bcast(&n, 1, MPI_INT, 0, comm);
	return n;
} 

/*---------------------------------------------------------------------
 * Function:  Build_blk_col_type
 * Purpose:   Build an MPI_Datatype that represents a block column of
 *            a matrix
 * In args:   n:  number of rows in the matrix and the block column
 *            loc_n = n/p:  number cols in the block column
 * Ret val:   blk_col_mpi_t:  MPI_Datatype that represents a block
 *            column
 */
MPI_Datatype Build_blk_col_type(int n, int loc_n) {
	MPI_Aint lb, extent;
	MPI_Datatype block_mpi_t;
	MPI_Datatype first_bc_mpi_t;
	MPI_Datatype blk_col_mpi_t;

	MPI_Type_contiguous(loc_n, MPI_INT, &block_mpi_t);
	MPI_Type_get_extent(block_mpi_t, &lb, &extent);

	MPI_Type_vector(n, loc_n, n, MPI_INT, &first_bc_mpi_t);
	MPI_Type_create_resized(first_bc_mpi_t, lb, extent,
			&blk_col_mpi_t);
	MPI_Type_commit(&blk_col_mpi_t);

	MPI_Type_free(&block_mpi_t);
	MPI_Type_free(&first_bc_mpi_t);

	return blk_col_mpi_t;
}  /* Build_blk_col_type */


void Read_matrix(char * path, int loc_mat[], int n, int loc_n, 
	MPI_Datatype blk_col_mpi_t, int my_rank, MPI_Comm comm, double *start_time) {
	int* mat = NULL, i, j;

	if (my_rank == 0) {
		FILE * file;
		int temp;
		file = fopen(path, "r");
		fscanf(file, "%d", &temp);
		mat = malloc(n*n*sizeof(int));
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
				fscanf(file, "%d", &mat[i*n + j]);
		fclose(file);
	}
	*start_time = (double)(clock());
	MPI_Scatter(mat, 1, blk_col_mpi_t, loc_mat, n*loc_n, MPI_INT, 0, comm);

	if (my_rank == 0) free(mat);
}

// save the distance information to the txt file
void Print_dists(int dist[], int n) {
	int v;
	FILE * file = fopen("parallel_dist.txt", "w");
	fprintf(file, "  v    dist 0->v\n");
	fprintf(file, "----   ---------\n");
						
	for (v = 1; v < n; v++)
		fprintf(file, "%3d       %4d\n", v, dist[v]);
	fclose(file);
} 

// save the path information to the txt file
void Print_paths(int pred[], int n) {
	int v, w, *path, count, i;
	
	FILE * file = fopen("parallel_path.txt", "w");
	
	path =  malloc(n*sizeof(int));

	fprintf(file, "  v     Path 0->v\n");
	fprintf(file, "----    ---------\n");
	for (v = 1; v < n; v++) {
		fprintf(file, "%3d:    ", v);
		count = 0;
		w = v;
		while (w != 0) {
			path[count] = w;
			count++;
			w = pred[w];
		}
		fprintf(file, "0 ");
		for (i = count-1; i >= 0; i--)
			fprintf(file, "%d ", path[i]);
		fprintf(file, "\n");
	}
	fclose(file);
	free(path);
}
