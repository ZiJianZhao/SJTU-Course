/* File:     dijkstra.c
 * Purpose:  Implement Dijkstra's algorithm for solving the single-source 
 *           shortest path problem:  find the length of the shortest path 
 *           between a specified vertex and all other vertices in a 
 *           directed graph.
 *
 * Input:    n, the number of vertices in the digraph
 *           mat, the adjacency matrix of the digraph
 * Output:   A list showing the cost of the shortest path
 *           from vertex 0 to every other vertex in the graph.
 *
 * Compile:  gcc -g -Wall -o dijkstra dijkstra.c
 * Run:      ./dijkstra
 *           For large matrices, put the matrix into a file with n as
 *           the first line and run with ./dijkstra < large_matrix
 *
 * Notes:
 * 1.  Edge lengths should be nonnegative.
 * 2.  The distance from v to w may not be the same as the distance from
 *     w to v.
 * 3.  If there is no edge between two vertices, the length is the constant
 *     INFINITY.  So input edge length should be substantially less than
 *     this constant.
 * 4.  The cost of travelling from a vertex to itself is 0.  So the adjacency
 *     matrix has zeroes on the main diagonal.
 * 5.  No error checking is done on the input.
 * 6.  The adjacency matrix is stored as a 1-dimensional array and subscripts
 *     are computed using the formula:  the entry in the ith row and jth
 *     column is mat[i*n + j]
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
const int INFINITY = 1000000;

int Read_n(char * path);
void Read_matrix(char * path, int mat[], int n);
void Print_matrix(int mat[], int n);
void Print_dists(int dist[], int n);
void Print_paths(int pred[], int n);
int Find_min_dist(int dist[], int known[], int n);
void Dijkstra(int mat[], int dist[], int pred[], int n);

int main(int argc, char *argv[]) {
	int  n;
	int *mat, *dist, *pred;
	clock_t start_t, end_t;
	if (argc != 2) {
		printf("The number of arguments is wrong. Your should enter the test file name.\n");
		return -1;
	}
	n = Read_n(argv[1]);
	if (n == -1) {
		return -1;
	}

	mat = malloc(n*n*sizeof(int));
	dist = malloc(n*sizeof(int));
	pred = malloc(n*sizeof(int));

	Read_matrix(argv[1], mat, n);

	start_t = clock();
	Dijkstra(mat, dist, pred, n);
	end_t = clock();
	double elapsed_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	printf("The elapsed time is: %lf.\n", elapsed_t);
	
	Print_dists(dist, n);
	Print_paths(pred, n);
	printf("You can see the distance from 0 in the 'serial_dist.txt' file.\n");
	printf("You can see the shortest path from 0 in the 'serial_path.txt' file.\n");
	free(mat);
	free(dist);
	free(pred);
	return 0;
}

int Read_n(char * path) {
	int n;
	FILE * file;
	file = fopen(path, "r");
	if (file != NULL) {
		printf("The file is sucessfully readed.\n");
		fscanf(file, "%d", &n);
		fclose(file);
		return n;
	} else {
		printf("The file doesn't exists or access is denied.\n");
		return -1;
	}
}  

void Read_matrix(char * path,  int mat[], int n) {
	int i, j;

	FILE * file;
	int temp;
	file = fopen(path, "r");
	fscanf(file, "%d", &temp);
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			fscanf(file, "%d", &mat[i*n + j]);
	fclose(file);
}  

void Print_matrix(int mat[], int n) {
	int i, j;

	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++)
			if (mat[i*n+j] == INFINITY)
				printf("i ");
			else
				printf("%d ", mat[i*n+j]);
		printf("\n");
	}
}  

/*-------------------------------------------------------------------
 * Function:    Dijkstra
 * Purpose:     Apply Dijkstra's algorithm to the matrix mat
 * In args:     n:  the number of vertices
 *              mat:  adjacency matrix for the graph
 * Out args:    dist:  dist[v] = distance 0 to v.
 *              pred:  pred[v] = predecessor of v on a 
 *                  shortest path 0->v.
 */
void Dijkstra(int mat[], int dist[], int pred[], int n) {
	int i, u, v, *known, new_dist;

	/* known[v] = true, if the shortest path 0->v is known */
	/* known[v] = false, otherwise                         */
	known = malloc(n*sizeof(int));

	/* Initialize d and p */
	dist[0] = 0; pred[0] = 0; known[0] = 1; 
	for (v = 1; v < n; v++) {
		dist[v] = mat[0*n + v];
		pred[v] = 0;
		known[v] = 0;
	}

#     ifdef DEBUG
		printf("i = 0\n");
		Print_dists(dist, n);
#     endif

	/* On each pass find an additional vertex */
	/* whose distance to 0 is known           */
	for (i = 1; i < n; i++) {
		u = Find_min_dist(dist, known, n);

		known[u] = 1;

		for (v = 1; v < n; v++) 
			if (!known[v]) {
				new_dist = dist[u] + mat[u*n + v];
				if (new_dist < dist[v]) {
					dist[v] = new_dist;
					pred[v] = u;
				}
			}

#     ifdef DEBUG
		printf("i = %d\n", i);
		Print_dists(dist, n);
#     endif
	} /* for i */

	free(known);
}  

int Find_min_dist(int dist[], int known[], int n) {
	int v, u, best_so_far = INFINITY;

	for (v = 1; v < n; v++)
		if (!known[v])
			if (dist[v] < best_so_far) {
				u = v;
				best_so_far = dist[v];
			}

	return u;
}  

void Print_dists(int dist[], int n) {
	int v;
	FILE * file = fopen("serial_dist.txt", "w");
	fprintf(file, "  v    dist 0->v\n");
	fprintf(file, "----   ---------\n");
						
	for (v = 1; v < n; v++)
		fprintf(file, "%3d       %4d\n", v, dist[v]);
	fclose(file);
} 

void Print_paths(int pred[], int n) {
	int v, w, *path, count, i;
	
	FILE * file = fopen("serial_path.txt", "w");
	
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