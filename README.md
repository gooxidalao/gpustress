Multi-GPU CUDA stress test tool.

build

编译：nvcc stressgpu.cu -o stress_gpu_multi -O2 -lm

usage

./stress_gpu_multi 120 256 8

运行120 秒，每元素 256 次计算，用 8 张 GPU
