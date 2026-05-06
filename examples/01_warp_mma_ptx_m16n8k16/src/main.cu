#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define SIZE 16
#define WARP_SIZE 32

std::string join_path(const std::string &base, const std::string &child) {
  if (base.empty()) {
    return child;
  }
  if (base.back() == '/') {
    return base + child;
  }
  return base + "/" + child;
}

std::string executable_dir() {
  char path_buf[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", path_buf, sizeof(path_buf) - 1);
  if (len <= 0) {
    return ".";
  }
  path_buf[len] = '\0';
  std::string full_path(path_buf);
  size_t slash = full_path.find_last_of('/');
  if (slash == std::string::npos) {
    return ".";
  }
  return full_path.substr(0, slash);
}

void print_usage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " [--input-dir <dir>] [--output-dir <dir>]\n";
}

void read_half(const std::string &file, std::vector<half> &v) {
  std::ifstream in(file);
  if (!in) {
    std::cerr << "Failed to open input file: " << file << "\n";
    std::exit(1);
  }
  float tmp;
  for (int i = 0; i < SIZE * SIZE; i++) {
    in >> tmp;
    if (!in) {
      std::cerr << "Invalid or insufficient data in input file: " << file << "\n";
      std::exit(1);
    }
    v[i] = __float2half(tmp);
  }
}

void write_float(const std::string &file, const std::vector<float> &v) {
  std::ofstream out(file);
  for (float x : v) {
    out << x << " ";
  }
}

__global__ void tensor_core_kernel(const half *, const half *, float *);

int main(int argc, char **argv) {
  std::string input_dir;
  std::string output_dir;

  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "--input-dir") {
      if (i + 1 >= argc) {
        print_usage(argv[0]);
        return 1;
      }
      input_dir = argv[++i];
    } else if (arg == "--output-dir") {
      if (i + 1 >= argc) {
        print_usage(argv[0]);
        return 1;
      }
      output_dir = argv[++i];
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  const std::string exe_dir = executable_dir();
  if (input_dir.empty()) {
    input_dir = join_path(exe_dir, "inputs");
  }
  if (output_dir.empty()) {
    output_dir = join_path(exe_dir, "outputs");
  }

  if (mkdir(input_dir.c_str(), 0755) != 0 && errno != EEXIST) {
    std::cerr << "Failed to create input directory: " << input_dir
              << ": " << std::strerror(errno) << "\n";
    return 1;
  }
  if (mkdir(output_dir.c_str(), 0755) != 0 && errno != EEXIST) {
    std::cerr << "Failed to create output directory: " << output_dir
              << ": " << std::strerror(errno) << "\n";
    return 1;
  }

  std::vector<half> A(SIZE * SIZE), B(SIZE * SIZE);
  std::vector<float> C(SIZE * SIZE, 0.0f);

  const std::string a_path = join_path(input_dir, "A.txt");
  const std::string b_path = join_path(input_dir, "B.txt");
  const std::string c_path = join_path(output_dir, "C_gpu.txt");

  read_half(a_path, A);
  read_half(b_path, B);

  half *dA = nullptr;
  half *dB = nullptr;
  float *dC = nullptr;

  cudaMalloc(&dA, sizeof(half) * SIZE * SIZE);
  cudaMalloc(&dB, sizeof(half) * SIZE * SIZE);
  cudaMalloc(&dC, sizeof(float) * SIZE * SIZE);

  cudaMemcpy(dA, A.data(), sizeof(half) * SIZE * SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), sizeof(half) * SIZE * SIZE, cudaMemcpyHostToDevice);

  tensor_core_kernel<<<1, WARP_SIZE>>>(dA, dB, dC);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  cudaMemcpy(C.data(), dC, sizeof(float) * SIZE * SIZE, cudaMemcpyDeviceToHost);
  write_float(c_path, C);

  std::cout << "Stored " << (SIZE * SIZE)
            << " values (16x16 row-major) in " << c_path << "\n";
  std::cout << "First row: ";
  for (int i = 0; i < SIZE; i++) {
    std::cout << C[i] << (i + 1 == SIZE ? '\n' : ' ');
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
