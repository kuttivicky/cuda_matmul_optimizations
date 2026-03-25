#include <stdio.h>
// #include <iostream.h>
#include <cuda_runtime.h>
#include <time.h>

#define A 2048

__global__ void matmul_naive(float* M, float* N, float* P, int width) {
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row<width && col<width){
        float Pvalue = 0.0f;
        for (int k = 0; k<width; k++){
            Pvalue += M[row*width + k] * N[k*width + col];
        }
        P[row*width + col] = Pvalue;
    }
}

void matmul_cpu(const float* M, const float* N, float* P, int width) {
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += M[row * width + k] * N[k * width + col];
            }
            P[row * width + col] = sum;
        }
    }
}

int main(){
    float *M , *N, *P, *P_cpu;
    float *d_a, *d_b, *d_c;

    size_t size = A * A * sizeof(float);

    M = (float*)malloc(size); 
    N = (float*)malloc(size);
    P = (float*)malloc(size);
    P_cpu = (float*)malloc(size);

    float element1 = 1.0f;
    float element2 = 1.0f;

    for(int i = 0; i<A*A; i++) M[i] = element1++;
    for(int i = 0; i<A*A; i++) N[i] = ++element2;

    // clock_t cpu_start = clock();
    // matmul_cpu(M, N, P_cpu, A);
    // clock_t cpu_end = clock();
    // double cpu_ms = 1000.0 * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    // printf("CPU time: %f ms\n", cpu_ms);

    int width = A;
    dim3 blockDim(8, 8);
    dim3 gridDim((width - 1) / 8 + 1, (width - 1) / 8 + 1);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, N, size, cudaMemcpyHostToDevice);

    // matmul_naive<<< gridDim, blockDim >>> (d_a, d_b, d_c, width);

    int runs = 100;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int i = 0; i < runs; i++)
    {
        matmul_naive<<< gridDim, blockDim >>> (d_a, d_b, d_c, width);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms,start,stop);

    printf("GPU avg time (%d runs): %f ms\n", runs, ms / runs);

    // printf("CPU sample P[0][0] = %f\n", P_cpu[0]);

    cudaMemcpy(P, d_c, size, cudaMemcpyDeviceToHost);
    printf("GPU sample P[0][0] = %f\n", P[0]);

    // for(int i =0; i<3; i++){
    //     for(int j = 0; j<3; j++){
    //         printf("P[%d][%d] = %f ", i, j, P[i*A + j]);
    //     }
    //     printf("\n");
    // }

    free(M);
    free(N);
    free(P);
    free(P_cpu);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}