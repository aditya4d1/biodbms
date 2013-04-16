/*------------------------------------------
 ---- Created By: Aditya Avinash Atluri ----
 ------- you are free to use any code ------
 ------- Submit any issues or errors -------
 -------------------------------------------*/

#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include"cuda_runtime.h"

__device__ __global__ void Mean2c(float *,float *);
__device__ __global__ void Meanfinald(float *);
__device__ __global__ void Std(float *,float *);
__device__ __global__ void Std1(float *,float *);

#define Block 512  	//Size of Array in Shared Memory and Number of threads in a block
#define Grid 8			//Size of Number of Blocks
#define Total 131072	//Total number of Samples per block
#define Width 4096		//Total number of Samples per Grid
#define Length 16384	//Total Size of Shared Memory
#define Max 1048576		//Total number of samples

int main(void){
	float *A,B[Max/Block];
	A=(float *)malloc(sizeof(float)*Max);
	for(int i=0;i<Max;i++){
		A[i]=i;
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
	dim3 dimBlock1(256,1);
	dim3 dimGrid1(Grid,1);
	dim3 dimBlock2(256,256,8);
	dim3 dimGrid2(8,1);
	for(int i=0;i<loop;i++){
		Mean2c<<<dimGrid,dimBlock>>>(Ad+(i*Width),Bd+(i*Grid));
	}
	Meanfinald<<<dimGrid1,dimBlock1>>>(Bd);
	for(int j=0;j<loop;j++){
		Std<<<dimGrid,dimBlock>>>(Ad+(j*Width),Bd);
	}
	Std1<<<dimGrid2,dimBlock2>>>(Ad,Bd);
	cudaMemcpy(B,Ad,8*sizek,cudaMemcpyDeviceToHost);
	for(int i=0;i<8;i++){
		printf("%f	\n",B[i]);
	}
	int quit;
	scanf("%d",&quit);
	return 0;
}


__device__ __global__ void Mean2c(float *Ad,float *Bd){
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
	__device__ __shared__ float Bs[256];
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	Bs[tx]=Bd[tx+bx*256];
		if(tx%8==0){
			Bs[tx]=(Bs[tx]+Bs[tx+1]+Bs[tx+2]+Bs[tx+3]+Bs[tx+4]+Bs[tx+5]+Bs[tx+6]+Bs[tx+7])/8;
		}
		if(tx%64==0){
			Bs[tx]=(Bs[tx]+Bs[tx+8]+Bs[tx+16]+Bs[tx+24]+Bs[tx+32]+Bs[tx+40]+Bs[tx+48]+Bs[tx+56])/8;
		}
		if(tx==0){
			Bs[tx]=(Bs[tx]+Bs[tx+64]+Bs[tx+128]+Bs[tx+192])/4;
		}
	Bd[bx]=Bs[0];
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

__device__ __global__ void Std1(float *Ad,float*Bd){
	__device__ __shared__ float As[256];
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int tz=threadIdx.z;
	int bx=blockIdx.x;
	Ad[tz+ty*8]=Ad[tz+ty*4096];
	As[tx]=Ad[tx+bx*256];
		if(tx%8==0){
			As[tx]=(As[tx]+As[tx+1]+As[tx+2]+As[tx+3]+As[tx+4]+As[tx+5]+As[tx+6]+As[tx+7]);
		}
		if(tx%64==0){
			As[tx]=(As[tx]+As[tx+8]+As[tx+16]+As[tx+24]+As[tx+32]+As[tx+40]+As[tx+48]+As[tx+56]);
		}
		if(tx==0){
			As[tx]=(As[tx]+As[tx+64]+As[tx+128]+As[tx+192]);
		}
	Ad[bx]=As[0];
	Ad[bx]=sqrt(Ad[bx]/Total);
}

//	Here, we have a block having 512 threads.
//	Each Grid has 32 Blocks. And, we have only 1 Grid. You know.!!
//	We use shared memory the total 16KB (16384B).
//	We divide it into 32 parts. As we have 32 Blocks.
//	Each Block now has 512B of Memory (128 of floats) (32*4*128)
