#include <iostream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
using namespace std;

void print_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) 
            cout<<matrix[i*cols+j]<<" ";
        cout<<endl;
    }
}

void cpu(float* matrix, float* transpose, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            transpose[j*rows+i] = matrix[i*cols+j];
}

__global__ void naive_gpu(float* matrix, float* transpose, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ((i >= rows) || (j >= cols)) {
        return;
    }
    int index_in = i * cols + j;
    int index_out = j * rows + i;
    transpose[index_out] = matrix[index_in];
}

const int BLOCK_SIZE_X = 32;
const int BLOCK_SIZE_Y = 32;
__global__ void shared_memory(float* matrix, float* transpose, int rows, int cols) {
    __shared__ float mat[BLOCK_SIZE_X][BLOCK_SIZE_Y];
    int bx = blockIdx.x * BLOCK_SIZE_X;
    int by = blockIdx.y * BLOCK_SIZE_Y;
    int i = by + threadIdx.y; int j = bx +threadIdx.x;
    int ti = bx + threadIdx.y; int tj = by + threadIdx.x;
    if ((i < rows) && (j < cols))
        mat[threadIdx.x][threadIdx.y] = matrix[i * cols + j];
    __syncthreads();
    if ((tj < rows) && (ti < cols))
        transpose[ti * rows + tj] = mat[threadIdx.y][threadIdx.x];
}

__global__ void no_bank_conflict(float * matrix, float * transpose, int rows, int cols) {
    __shared__ float mat[BLOCK_SIZE_X][BLOCK_SIZE_Y + 1];
    int bx = blockIdx.x * BLOCK_SIZE_X;
    int by = blockIdx.y * BLOCK_SIZE_Y;
    int i = by + threadIdx.y; int j = bx +threadIdx.x;
    int ti = bx + threadIdx.y; int tj = by + threadIdx.x;
    if ((i < rows) && (j < cols))
        mat[threadIdx.x][threadIdx.y] = matrix[i * cols + j];
    __syncthreads();
    if ((tj < rows) && (ti < cols))
        transpose[ti * rows + tj] = mat[threadIdx.y][threadIdx.x];
}


const int TILE = 32;
const int SIDE = 8;
__global__ void loop_unrolled(float * matrix, float * transpose, int rows, int cols) {
    __shared__ float mat[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
#pragma unroll
    for(int k = 0; k < TILE ; k += SIDE) {
        if(x < rows && y + k < cols)
            mat[threadIdx.y+k][threadIdx.x] = matrix[(y + k)*rows + x];
    }
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
#pragma unroll
    for(int k = 0; k < TILE; k += SIDE) {
        if(x < cols && y + k < rows)
            transpose[(y + k) * cols + x] = mat[threadIdx.x][threadIdx.y+k];
    }
}


int main(int argc, char *argv[])
{
    int rows = 4000;
    int cols = 4000;
    string ver = "cpu";
    if (argc == 1) {
        ;
    } else if (argc == 3) {
        string str1 = argv[1];
        rows = atoi(str1.c_str());
        string str2 = argv[2];
        cols = atoi(str2.c_str());
    } else if (argc == 4) {
        string str1 = argv[1];
        rows = atoi(str1.c_str());
        string str2 = argv[2];
        cols = atoi(str2.c_str());
        ver = argv[3];
        if ((ver != "cpu") && (ver != "naive_gpu") && (ver != "shared_memory")\
         && (ver != "no_bank_conflict") && (ver != "loop_unrolled")) {
            cout<<"Wrong parameters of program version, you can only choose from following:";
            cout<<" cpu, naive_gpu, shared_memory, no_bank_conflict, loop_unrolled."<<endl;
            return 1;
        }
    } else {
        cout<<"Wrong number of parameters."<<endl;
        return 1;
    }
    if ((rows <= 0) || (cols <= 0)) {
        cout<<"Wrong parameters of the the matrix dimension, should be two positive numbers.";
        cout<<endl;
        return 1;
    }
    cout<<"Matrix Dimension: ("<<rows<<", "<<cols<<")"<<endl;
    cout<<"Program Version: "<<ver<<endl;
    
    float* matrix = new float[rows*cols];
    float* transpose = new float[rows*cols]; 
    srand((unsigned)time(NULL));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i*cols+j] = rand();
    
    if (ver == "cpu") {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        for (int iter = 0; iter < 100; ++iter)
            cpu(matrix, transpose, rows, cols);    
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime=0;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Elapsed time：%f <ms>\n", elapsedTime);    
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    else {
        float *dev_matrix, *dev_transpose;         
        cudaMalloc((void**)&dev_matrix, rows*cols*sizeof(float));  
        cudaMalloc((void**)&dev_transpose, rows*cols*sizeof(float));
        cudaMemcpy(dev_matrix, matrix, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        dim3 dimGrad(cols/32+1,rows/32+1,1);
        dim3 dimBlock(32,32,1);
        if (ver == "naive_gpu") {
            naive_gpu<<<dimGrad, dimBlock>>>(dev_matrix, dev_transpose, rows, cols);
        } else if (ver == "shared_memory") {
            shared_memory<<<dimGrad, dimBlock>>>(dev_matrix, dev_transpose, rows, cols);
        } else if (ver == "no_bank_conflict") {
            no_bank_conflict<<<dimGrad, dimBlock>>>(dev_matrix, dev_transpose, rows, cols);
        } else {
            dim3 dimGrad(cols/32+1,rows/32+1,1);
            dim3 dimBlock(32,8,1);
            loop_unrolled<<<dimGrad, dimBlock>>>(dev_matrix, dev_transpose, cols, rows); 
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime=0;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Elapsed time：%f <ms>\n", elapsedTime);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaMemcpy(transpose, dev_transpose, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
        
        //print_matrix(matrix, rows, cols);
        //printf("==========================\n");
        //print_matrix(transpose, cols, rows);
        
        cudaFree(dev_matrix);
        cudaFree(dev_transpose);
    }
    free(matrix);
    free(transpose);
    return 0;
}
