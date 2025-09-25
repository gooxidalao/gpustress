#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

__global__ void mem_compute_stress_kernel(float *a, float *b, int n, int compute_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx];

        // 大量单精度计算
        #pragma unroll 32
        for (int i = 0; i < compute_iters; i++) {
            val = val * 1.000001f + 0.000001f;  // FMA
            val = sinf(val);                    // 三角函数
        }

        b[idx] = val;  // 防止被优化
    }
}

void run_on_device(int dev, int duration, int compute_iters) {
    cudaSetDevice(dev);

    const int N = 1 << 27;  // ~512MB
    const size_t size = N * sizeof(float);

    float *d_a, *d_b;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    float *h_a = new float[N];
    for (int i = 0; i < N; i++) h_a[i] = static_cast<float>(i);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    dim3 block(1024);
    dim3 grid((N + block.x - 1) / block.x);

    std::cout << "GPU " << dev << " stress test started for "
              << duration << " sec ..." << std::endl;

    auto start = std::chrono::steady_clock::now();

    while (true) {
        mem_compute_stress_kernel<<<grid, block>>>(d_a, d_b, N, compute_iters);
        cudaDeviceSynchronize();

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() >= duration) {
            break;
        }
    }

    std::cout << "GPU " << dev << " stress test finished." << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <duration_sec> <compute_iters> <gpu_count>" << std::endl;
        return 1;
    }

    int duration = std::stoi(argv[1]);      // 测试时长（秒）
    int compute_iters = std::stoi(argv[2]); // 每元素的额外计算次数
    int gpu_count_req = std::stoi(argv[3]); // 期望使用 GPU 数量

    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return 1;
    }

    int gpu_count = std::min(std::min(gpu_count_req, device_count), 8);

    std::cout << "Detected " << device_count << " GPUs, using "
              << gpu_count << " for stress test." << std::endl;

    std::vector<std::thread> threads;
    for (int i = 0; i < gpu_count; i++) {
        threads.emplace_back(run_on_device, i, duration, compute_iters);
    }

    for (auto &t : threads) {
        t.join();
    }

    std::cout << "All GPUs finished stress test." << std::endl;
    return 0;
}

