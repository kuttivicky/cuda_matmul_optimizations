#include <stdio.h>
// #include <iostream.h>
#include <cuda_runtime.h>
#include <time.h>

#define A 2048

// change naive to interchage row and col to get bad performance
// do warmup in coalesced

__global__ void matmul_coalesced(float* M, float* N_T, float* P, int width) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row<width && col<width){
        float Pvalue = 0.0f;
        for (int k = 0; k<width; k++){
            Pvalue += M[row*width + k] * N_T[k*width + col];  //[k*width + col]
        }
        P[row*width + col] = Pvalue;
    }
}

void transpose(const float* src, float* dst, int width)
{
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j < width; j++)
        {
            dst[j * width + i] = src[i * width + j];
        }
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

    // transpose(N, N_T, A);

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
        matmul_coalesced<<<gridDim, blockDim>>>(d_a, d_b, d_c, width);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for(int i = 0; i < runs; i++)
    {
        matmul_coalesced<<< gridDim, blockDim >>> (d_a, d_b, d_c, width);
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