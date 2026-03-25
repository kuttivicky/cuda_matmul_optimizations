#include <stdio.h>
// #include <iostream.h>
#include <cuda_runtime.h>
#include <time.h>

#define A 2048
#define TILE 16

// change naive to interchage row and col to get bad performance
// do warmup in coalesced

__global__ void matmul_tiled(float* M, float* N, float* P, int width) {

    __shared__ float Ms[TILE][TILE];
    __shared__ float Ns[TILE][TILE];

    int row = TILE * blockIdx.y + threadIdx.y;
    int col = TILE * blockIdx.x + threadIdx.x;

    float Pvalue = 0.0f;

    //tile index t loop over tiles
    for (int t = 0; t < (width + TILE - 1)/TILE; t++){
        //loading...
        if (row<width && t * TILE + threadIdx.x < width){
            Ms[threadIdx.y][threadIdx.x] = M[row * width + ( t * TILE + threadIdx.x)];
        }
        else{
            Ms[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col<width && t * TILE + threadIdx.y < width){
            Ns[threadIdx.y][threadIdx.x] = N[( t * TILE + threadIdx.y) * width + col];
        }
        else{
            Ns[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for ( int k = 0; k<TILE; k++){
            Pvalue += Ms[threadIdx.y][k] * Ns[k][threadIdx.x]; 
        }
        __syncthreads();
    }

    if(row<width && col<width){
        P[row * width + col] = Pvalue;
    }
}

int main(){
    float *M , *N, *P;
    float *d_a, *d_b, *d_c;

    size_t size = A * A * sizeof(float);

    M = (float*)malloc(size); 
    N = (float*)malloc(size);
    P = (float*)malloc(size);
    
    float element1 = 1.0f;
    float element2 = 1.0f;

    for(int i = 0; i<A*A; i++) M[i] = element1++;
    for(int i = 0; i<A*A; i++) N[i] = ++element2;

    int width = A;
    dim3 blockDim(16, 16);
    dim3 gridDim((width - 1) / 16 + 1, (width - 1) / 16 + 1);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, N, size, cudaMemcpyHostToDevice); // transpose

    // matmul_naive<<< gridDim, blockDim >>> (d_a, d_b, d_c, width);

    int runs = 100;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 5; i++) { // warmup
        matmul_tiled<<<gridDim, blockDim>>>(d_a, d_b, d_c, width);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i = 0; i < runs; i++)
    {
        matmul_tiled<<< gridDim, blockDim >>> (d_a, d_b, d_c, width);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms,start,stop);

    printf("GPU avg time (%d runs): %f ms\n", runs, ms / runs);

    cudaMemcpy(P, d_c, size, cudaMemcpyDeviceToHost);

    printf("P[0][0]: %f", P[0]);

    free(M);
    free(N);
    free(P);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}