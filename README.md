# cuda_matmul_optimizations
I have tried to optimize matrix multiplications as close as possible to cuBLAS

## 📊 Performance Results

| Kernel        | Time (ms) | GFLOPS |
|--------------|----------|--------|
| Naive        | 195 ms   | 88     |
| Coalesced    | 69 ms    | 249    |
| Tiled        | 46 ms    | 373    |
| cuBLAS       | 10 ms    | 1718   |

---
