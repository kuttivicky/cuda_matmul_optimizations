#include <stdio.h>
// #include <iostream.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cublas_v2.h>

#define A 2048
#define TILE 16

// change naive to interchage row and col to get bad performance
// do warmup in coalesced


int main(){
    cublasHandle_t handle; //column major
    cublasCreate(&handle);

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


    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, N, size, cudaMemcpyHostToDevice); 

    float alpha = 1.0f;
    float beta = 0.0f;

    int runs = 100;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 5; i++) { // warmup
        cublasSgemm_v2(handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       A,A,A,
                       &alpha,
                       d_b, A,
                       d_a, A,
                       &beta,
                       d_c, A);
    }

    cudaEventRecord(start);

    for(int i = 0; i < runs; i++)
    {
        cublasSgemm_v2(handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            A,A,A,
            &alpha,
            d_b, A,
            d_a, A,
            &beta,
            d_c, A);
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