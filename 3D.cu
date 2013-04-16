/*------------------------------------------
 ---- Created By: Aditya Avinash Atluri ----
 ------- you are free to use any code ------
 ------- Submit any issues or errors -------
 -------------------------------------------*/

#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include"cuda_runtime.h"

__device__ __global__ void Mean3d(float *,float *);
__device__ __global__ void Meanfinald(float *);
__device__ __global__ void Meanfinal1d(float *);
__device__ __global__ void Std(float *,float *);
__device__ __global__ void Std1(float *);
__device__ __global__ void Std4(float *);
__device__ __global__ void Std2(float *,float *);
__device__ __global__ void Std3(float *);

#define Block 512  	//Size of Array in Shared Memory and Number of threads in a block
#define Grid 8			//Size of Number of Blocks
#define Total 1048576	//Total number of Samples per Cluster
#define Dimen 128		//Number of Clusters
#define Max 134217728	//Total number of samples
#define Width 4096		//Total number of Samples per Grid
#define Length 16384	//Total Size of Shared Memory

int main(void){
	float *A,*B;
	B=(float *)malloc(sizeof(float)*Max/Block);
	A=(float *)malloc(sizeof(float)*Max);
	for(int i=0;i<Max;i++){
		A[i]=(i+1)*0.0001;
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
	dim3 dimBlock1(Block,8);
	dim3 dimGrid1(Grid,2);
	dim3 dimBlock2(2,64);
	dim3 dimGrid2(1,1);
	dim3 dimBlock3(256,1);
	dim3 dimGrid3(128,1);
	dim3 dimBlock4(256,1);
	dim3 dimGrid4(16,1);
	dim3 dimBlock5(1,1);
	dim3 dimGrid5(128,1);
	dim3 dimBlock6(16,1);
	dim3 dimGrid6(8,1);
	for(int i=0;i<loop;i++){
		Mean334d<<<dimGrid,dimBlock>>>(Ad+(i*Width),Bd+(i*Grid));
	}
	for(int i=0;i<64;i++){
		Meanfinald<<<dimGrid1,dimBlock1>>>(Bd+(i*4096));
	}
	Meanfinal1d<<<dimGrid2,dimBlock2>>>(Bd);
	for(int j=0;j<loop;j++){
		Std<<<dimGrid,dimBlock>>>(Ad+(j*Width));
	}
	Std3<<<dimGrid3,dimBlock3>>>(Ad);
	for(int i=0;i<8;i++){
		Std1<<<dimGrid4,dimBlock4>>>(Ad+i*4096);
	}
	Std4<<<dimGrid6,dimBlock6>>>(Ad);
	Std2<<<dimGrid5,dimBlock5>>>(Ad,Bd);
	cudaMemcpy(A,Ad,128*sizek,cudaMemcpyDeviceToHost);
	for(int i=0;i<128;i++){
		printf("%f	--	%d\n",A[i],i);
	}
	int quit;
	scanf("%d",&quit);
	return 0;
}


__device__ __global__ void Mean3d(float *Ad,float *Bd){
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
	__device__ __shared__ float Bs[Block];
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	Bs[tx]=Bd[tx+bx*Block];
		if(tx%8==0){
			Bs[tx]=(Bs[tx]+Bs[tx+1]+Bs[tx+2]+Bs[tx+3]+Bs[tx+4]+Bs[tx+5]+Bs[tx+6]+Bs[tx+7])/8;
		}
		if(tx%64==0){
			Bs[tx]=(Bs[tx]+Bs[tx+8]+Bs[tx+16]+Bs[tx+24]+Bs[tx+32]+Bs[tx+40]+Bs[tx+48]+Bs[tx+56])/8;
		}
		if(tx==0){
			Bs[tx]=(Bs[tx]+Bs[tx+64]+Bs[tx+128]+Bs[tx+192]+Bs[tx+256]+Bs[tx+320]+Bs[tx+384]+Bs[tx+448])/8;
		}
	Bd[bx]=Bs[0];
	int ty=threadIdx.y;
	int by=blockIdx.y;
	__device__ __shared__ float Cs[8];
	Cs[ty]=Bd[ty+by*8];
	if(ty==0){
		Cs[ty]=(Cs[ty]+Cs[ty+1]+Cs[ty+2]+Cs[ty+3]+Cs[ty+4]+Cs[ty+5]+Cs[ty+6]+Cs[ty+7])/8;
	}
	Bd[by]=Cs[0];
}


__device__ __global__ void Meanfinal1d(float *Bd){
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	Bd[tx+ty*2]=Bd[tx+ty*4096];
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
	Ad[0]=(Ad[0]+Ad[1]+Ad[2]+Ad[3]+Ad[4]+Ad[5]+Ad[6]+Ad[7])/8;
}


__device__ __global__ void Std1(float *Ad){
	__device__ __shared__ float As[256];
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	As[tx]=Ad[tx+bx*256];
	if(tx%8==0){
		As[tx]=(As[tx]+As[tx+1]+As[tx+2]+As[tx+3]+As[tx+4]+As[tx+5]+As[tx+6]+As[tx+7]);
	}
	if(tx%64==0){
		As[tx]=(As[tx]+As[tx+8]+As[tx+16]+As[tx+24]+As[tx+32]+As[tx+40]+As[tx+48]+As[tx+56]);
	}
	if(tx%256==0){
		As[tx]=(As[tx]+As[tx+64]+As[tx+128]+As[tx+192]);
	}
	Ad[bx]=As[0];
}


__device__ __global__ void Std3(float *Ad){
	int tx=threadIdx.x;
	int by=blockIdx.y;
	int bx=blockIdx.x;
	Ad[tx+bx*256]=Ad[tx*4096+bx*1048576];
}


__device__ __global__ void Std4(float *Ad){
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	Ad[tx+bx*16]=Ad[tx+bx*4096];
}


__device__ __global__ void Std2(float *Ad,float *Bd){
	__device__ __shared__ float As[1],Bs[1];
	int bx=blockIdx.x;
	As[0]=Ad[bx];
	Bs[0]=Bd[bx];
	As[0]=sqrt(As[0]/Total);
	Ad[bx]=As[0];
}


//	Here, we have a block having 512 threads.
//	Each Grid has 32 Blocks. And, we have only 1 Grid. You know.!!
//	We use shared memory the total 16KB (16384B).
//	We divide it into 32 parts. As we have 32 Blocks.
//	Each Block now has 512B of Memory (128 of floats) (32*4*128)
