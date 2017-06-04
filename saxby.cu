/*
* Compile: nvcc -o saxby saxby.cu
* Run: ./saxby
*/
#include <stdio.h>
__global__ void
daxbyAdd(const float *A, const float *B, float *C, float x,int numElements){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < numElements){
		C[i] = A[i]* x + B[i];
	}
}
int main (void){
	int N = 1<<20;
	float *x, *y, *z, *d_x, *d_y, *d_z;
	x = (float*) malloc(N*sizeof(float));
	y = (float*) malloc(N*sizeof(float));
	z = (float*) malloc(N*sizeof(float));

	cudaMalloc(&d_x, N*sizeof(float));
	cudaMalloc(&d_y, N*sizeof(float));
	cudaMalloc(&d_z, N*sizeof(float));
	
	for(int i = 0; i < N; i++){
		x[i] = 1.0f;
		y[i] = 1.0f;
		//z[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_z, z, N*sizeof(float), cudaMemcpyHostToDevice);

	daxbyAdd<<<(N+255)/256, 256>>>(d_x, d_y, d_z, 2.0f, N);

	cudaMemcpy(z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for(int i = 0; i < N; i++) {
		maxError = max(maxError, abs(y[i] - 3.0f));
	}
	printf("Max Error: %f\n", maxError);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	free(x);
	free(y);
	free(z);
}
