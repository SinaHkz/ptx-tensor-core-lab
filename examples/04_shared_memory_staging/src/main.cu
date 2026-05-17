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

#define M 16
#define WARP_SIZE 32

// Stage-04 delta vs stage 03:
//   - Math path stays the same (same mma.sync), but kernel now stages
//     A/B tiles in shared memory and accumulates into C with C += partial.
//   - Because C is accumulated (not overwritten) inside kernel, device C
//     buffer must be explicitly zero-initialized before launch.

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

std::vector<half> read_half_all(const std::string &file) {
  std::ifstream in(file);
  if (!in) {
    std::cerr << "Failed to open input file: " << file << "\n";
    std::exit(1);
  }

  std::vector<half> out;
  float tmp;
  while (in >> tmp) {
    out.push_back(__float2half(tmp));
  }
  if (out.empty()) {
    std::cerr << "Input file is empty: " << file << "\n";
    std::exit(1);
  }
  return out;
}

void write_float(const std::string &file, const std::vector<float> &v) {
  std::ofstream out(file);
  for (float x : v) {
    out << x << " ";
  }
  out << "\n";
}

__global__ void tensor_core_kernel(const half *, const half *, float *, int, int);

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

  const std::string a_path = join_path(input_dir, "A.txt");
  const std::string b_path = join_path(input_dir, "B.txt");
  const std::string c_path = join_path(output_dir, "C_gpu.txt");

  // Same dynamic shape inference as stage 03:
  //   A is MxK (M fixed to 16), B is KxN.
  const std::vector<half> A = read_half_all(a_path);
  if (A.size() % M != 0) {
    std::cerr << "Invalid A size: " << A.size()
              << " values is not divisible by M=" << M << "\n";
    return 1;
  }
  const int K = static_cast<int>(A.size() / M);

  const std::vector<half> B = read_half_all(b_path);
  if (B.size() % K != 0) {
    std::cerr << "Invalid B size: " << B.size()
              << " values is not divisible by K=" << K << "\n";
    return 1;
  }
  const int N = static_cast<int>(B.size() / K);

  // Tile-alignment constraints for m16n8k16.
  if (K % 16 != 0) {
    std::cerr << "Stage 04 requires K to be a multiple of 16. Got K=" << K << "\n";
    return 1;
  }
  if (N % 8 != 0) {
    std::cerr << "Stage 04 requires N to be a multiple of 8. Got N=" << N << "\n";
    return 1;
  }

  std::vector<float> C(M * N, 0.0f);

  half *dA = nullptr;
  half *dB = nullptr;
  float *dC = nullptr;

  cudaMalloc(&dA, sizeof(half) * A.size());
  cudaMalloc(&dB, sizeof(half) * B.size());
  cudaMalloc(&dC, sizeof(float) * C.size());

  cudaMemcpy(dA, A.data(), sizeof(half) * A.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), sizeof(half) * B.size(), cudaMemcpyHostToDevice);
  // Required for stage-04 accumulation path: kernel does C += d_tile.
  cudaMemset(dC, 0, sizeof(float) * C.size());

  tensor_core_kernel<<<1, WARP_SIZE>>>(dA, dB, dC, K, N);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
    return 1;
  }

  cudaMemcpy(C.data(), dC, sizeof(float) * C.size(), cudaMemcpyDeviceToHost);
  write_float(c_path, C);

  std::cout << "Stored " << C.size() << " values (" << M << "x" << N
            << " row-major, K=" << K << ") in " << c_path << "\n";
  std::cout << "First row: ";
  for (int i = 0; i < N; i++) {
    std::cout << C[i] << (i + 1 == N ? '\n' : ' ');
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
