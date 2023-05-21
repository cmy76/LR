#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.cuh"

void sumMatrix2D_CPU(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
  float * a=MatA;
  float * b=MatB;
  float * c=MatC;
  for(int j=0;j<ny;j++)
  {
    for(int i=0;i<nx;i++)
    {
      c[i]=a[i]+b[i];
    }
    c+=nx;
    b+=nx;
    a+=nx;
  }
}
__global__ void sumMatrix(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=ix+iy*ny;
    if (ix<nx && iy<ny)
    {
      MatC[idx]=MatA[idx]+MatB[idx];
    }
}

__global__ void multiplyMatrix(float * MatA,float * MatB,float * MatC,int B_col,int A_row, int common){
	int j=threadIdx.x+blockDim.x*blockIdx.x;
    int i=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=i*B_col+j;
	if(j<B_col && i<A_row)
	for(int k=0;k<common;k++){
		MatC[idx] += MatA[i*common+k] * MatB[k*B_col+j];
	}
}

void matrixMultiply(float * MatA, float * MatB, float * MatRes, int A_row, int A_col, int B_col){
	for(int i=0;i<A_row;i++){
		for(int j=0;j<B_col;j++){
			for(int k=0;k<A_col;k++){
				MatRes[i*A_col+j] += MatA[i*A_col+k] * MatB[k*B_col+j];
			}
		}
	}
}

void init_memory(float * Mat, int n_Element);

int main(int argc,char** argv)
{
	//printf("strating...\n");
	//initDevice(0);
	int nx=1<<9;
	int ny=1<<9;
	int nxy=nx*ny;
  	int nBytes=nxy*sizeof(float);

  //Malloc
	float* A_host=(float*)malloc(nBytes);
	float* B_host=(float*)malloc(nBytes);
	float* C_host=(float*)malloc(nBytes);
	float* A_from_gpu=(float*)malloc(nBytes);
	float* B_from_gpu=(float*)malloc(nBytes);
	float* C_from_gpu=(float*)malloc(nBytes);
	init_memory(A_host,nxy);
	init_memory(B_host,nxy);

  //cudaMalloc
	float *A_dev=NULL;
	float *B_dev=NULL;
	float *C_dev=NULL;
	CHECK(cudaMalloc((void**)&A_dev,nBytes));
	CHECK(cudaMalloc((void**)&B_dev,nBytes));
	CHECK(cudaMalloc((void**)&C_dev,nBytes));


	CHECK(cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(B_dev,B_host,nBytes,cudaMemcpyHostToDevice));

	int dimx=argc>2?atoi(argv[1]):32;
	int dimy=argc>2?atoi(argv[2]):32;

	double iStart,iElaps;
  // cpu compute
	iStart=cpuSecond();
	matrixMultiply(A_host,B_host,C_host,nx,ny,nx);

	printf("CPU test:\n");
	iElaps=cpuSecond()-iStart;
	printf("CPU Execution Time elapsed %f msec\n",iElaps*1000);
	//warm up
	// 2d block and 2d grid
	dim3 block_0(32,32);
	dim3 grid_0((nx-1)/block_0.x+1,(ny-1)/block_0.y+1);
	iStart=cpuSecond();
	multiplyMatrix<<<grid_0,block_0>>>(A_dev,B_dev,C_dev,nx,ny,ny);
	CHECK(cudaDeviceSynchronize());
	iElaps=cpuSecond()-iStart;
	printf("GPU Execution Time elapsed %f msec\n",iElaps*1000);
	CHECK(cudaMemcpy(C_from_gpu,C_dev,nBytes,cudaMemcpyDeviceToHost));
	checkResult(C_host,C_from_gpu,nxy);
	printf("Warm Up \n");

	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);
	free(A_host);
	free(B_host);
	free(C_host);
	free(C_from_gpu);
	cudaDeviceReset();
	return 0;
}


void init_memory(float * Mat, int n_Element){
	for(int i=0;i<n_Element;i++){
		Mat[i] = i+1;
	}
}

