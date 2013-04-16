/*------------------------------------------
 ---- Created By: Aditya Avinash Atluri ----
 ------- you are free to use any code ------
 ------- Submit any issues or errors -------
 -------------------------------------------*/

#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include"cuda_runtime.h"

__device__ __global__ void Mean3b(float *,float *);
__device__ __global__ void Meanfinald(float *);
__device__ __global__ void Meanfinal1d(float *);
__device__ __global__ void Std(float *,float *);
__device__ __global__ void Std1(float *);
__device__ __global__ void Std2(float *,float *);

#define Block 512  	//Size of Array in Shared Memory and Number of threads in a block
#define Grid 8			//Size of Number of Blocks
#define Dimen 128		//Number of Clusters
#define Total 16384		//Total number of Samples per Cluster
#define Max 2097152		//Total number of Samples
#define Width 4096		//Total number of Samples per Grid
#define Length 16384	//Total Size of Shared Memory

int main(void){
	float *A,B[Max/Block];
	A=(float *)malloc(Max*sizeof(float));
	for(int i=0;i<Max;i++){
		A[i]=(0.001)*i;
	}
	for(int i=0;i<(Max/Block);i++){
		B[i]=0;
	}
	float *Ad,*Bd;
	int size=Max*sizeof(float);
	int sizeb=(Max/Block)*sizeof(float);
	int sizek=sizeof(float);
	int loop=(Max/Width);
	cudaMalloc((void**)&Ad,size);
	cudaMalloc((void**)&Bd,sizeb);
	cudaMemcpy(Ad,A,size,cudaMemcpyHostToDevice);
	cudaMemcpy(Bd,B,sizeb,cudaMemcpyHostToDevice);
	dim3 dimBlock(Block,1);
	dim3 dimGrid(Grid,1);
	dim3 dimBlock1(32,1);
	dim3 dimGrid1(32,1);
	dim3 dimBlock2(8,1);
	dim3 dimGrid2(32,1);
	dim3 dimBlock3(32,1);
	dim3 dimGrid3(4,1);
	for(int i=0;i<loop;i++){
		Mean3b<<<dimGrid,dimBlock>>>(Ad+(i*Width),Bd+(i*Grid));
	}
	for(int i=0;i<4;i++){
		Meanfinald<<<dimGrid1,dimBlock1>>>(Bd+(i*1024));
	}
	Meanfinal1d<<<dimGrid2,dimBlock2>>>(Bd);
	for(int j=0;j<loop;j++){
		Std<<<dimGrid,dimBlock>>>(Ad+(j*Width),Bd);
	}
	for(int i=0;i<4;i++){
		Std1<<<dimGrid2,dimBlock2>>>(Ad+(i*524288));
	}
	Std2<<<dimGrid3,dimBlock3>>>(Ad,Bd);
	cudaMemcpy(A,Ad,128*sizek,cudaMemcpyDeviceToHost);
	for(int i=0;i<128;i++){
		printf("%f	-- SD	%d\n",A[i],i);
	}
	int quit;
	scanf("%d",&quit);
	return 0;
}


__device__ __global__ void Mean3b(float *Ad,float *Bd){
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	__device__ __shared__ float As[Block];
		As[tx]=Ad[tx+(bx*Block)];
		if(tx%8==0){
			As[tx]=(As[tx]+As[tx+1]+As[tx+2]+As[tx+3]+As[tx+4]+As[tx+5]+As[tx+6]+As[tx+7])/8;
		}
		if(tx%64==0){
			As[tx]=(As[tx]+As[tx+8]+As[tx+16]+As[tx+24]+As[tx+32]+As[tx+40]+As[tx+48]+As[tx+56])/8;
		}
		if(tx==0){
			As[tx]=(As[tx]+As[tx+64]+As[tx+128]+As[tx+192]+As[tx+256]+As[tx+320]+As[tx+384]+As[tx+448])/8;
		}
		Bd[bx]=As[0];
}

__device__ __global__ void Meanfinald(float *Bd){
	__device__ __shared__ float Bs[32];
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	Bs[tx]=Bd[tx+bx*32];
	if(tx%8==0){
		Bs[tx]=(Bs[tx]+Bs[tx+1]+Bs[tx+2]+Bs[tx+3]+Bs[tx+4]+Bs[tx+5]+Bs[tx+6]+Bs[tx+7])/8;
	}
	if(tx%32==0){
		Bs[tx]=(Bs[tx]+Bs[tx+8]+Bs[tx+16]+Bs[tx+24])/4;
	}
	Bd[bx]=Bs[0];
}

__device__ __global__ void Meanfinal1d(float *Bd){
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	Bd[tx+bx*32]=Bd[tx+bx*1024];
}

__device__ __global__ void Std(float *Ad,float *Bd){
	__device__ __shared__ float As[Block];
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	As[tx]=Ad[tx+(bx*Block)];
	As[tx]=As[tx]-Bd[0];
	if(tx%8==0){
			As[tx]=(As[tx]*As[tx])+(As[tx+1]*As[tx+1])+(As[tx+2]*As[tx+2])+(As[tx+3]*As[tx+3])+(As[tx+4]*As[tx+4])+(As[tx+5]*As[tx+5])+(As[tx+6]*As[tx+6])+(As[tx+7]*As[tx+7]);
	}
	if(tx%64==0){
			As[tx]=(As[tx]+As[tx+8]+As[tx+16]+As[tx+24]+As[tx+32]+As[tx+40]+As[tx+48]+As[tx+56]);
	}
	if(tx==0){
		As[tx]=(As[tx]+As[tx+64]+As[tx+128]+As[tx+192]+As[tx+256]+As[tx+320]+As[tx+384]+As[tx+448]);
	}
	Ad[bx]=As[0];
}

__device__ __global__ void Std1(float *Ad){
	__device__ __shared__ float As[32];
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	for(int i=0;i<4;i++){
		As[tx+i*8]=Ad[tx+i*4096+bx*16384];
	}
	if(tx%8==0){
		As[tx]=(As[tx]+As[tx+1]+As[tx+2]+As[tx+3]+As[tx+4]+As[tx+5]+As[tx+6]+As[tx+7]);
	}
	if(tx%32==0){
		As[tx]=(As[tx]+As[tx+8]+As[tx+16]+As[tx+24]);
	}
	Ad[bx]=As[0];
}


__device__ __global__ void Std2(float *Ad,float*Bd){
	__device__ __shared__ float As[128],Bs[128];
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	As[tx]=Ad[tx+bx*524288];
	Bs[tx]=Bd[tx+bx*32];
	{
		As[tx]=sqrt(As[tx]/Total);
	}
	Ad[tx+bx*32]=As[tx];
}

//	Here, we have a block having 512 threads.
//	Each Grid has 32 Blocks. And, we have only 1 Grid. You know.!!
//	We use shared memory the total 16KB (16384B).
//	We divide it into 32 parts. As we have 32 Blocks.
//	Each Block now has 512B of Memory (128 of floats) (32*4*128)
