// #include <iostream>
// #include <cuda_runtime.h>

// // CUDA 错误检查宏
// #define CUDA_CHECK(err) { \
//     cudaError_t error = err; \
//     if (error != cudaSuccess) { \
//         fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
//         exit(EXIT_FAILURE); \
//     } \
// }

// // 1. 核函数定义
// // 在 GPU 上执行向量加法
// __global__ void vectorAddKernel(const float *A, const float *B, float *C, int N) {
//     // 2. 线程组织：计算全局线程ID
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     // 边界检查，防止数组越界
//     if (tid < N) {
//         C[tid] = A[tid] + B[tid];
//     }
// }

// int main() {
//     int N = 1 << 20; // 1048576 个元素
//     size_t size = N * sizeof(float);

//     // 在主机上分配内存
//     float *h_A = new float[N];
//     float *h_B = new float[N];
//     float *h_C = new float[N];

//     // 初始化主机数据
//     for (int i = 0; i < N; ++i) {
//         h_A[i] = static_cast<float>(rand()) / RAND_MAX;
//         h_B[i] = static_cast<float>(rand()) / RAND_MAX;
//     }

//     // 3. & 4. GPU 内存管理与数据传输
//     float *d_A, *d_B, *d_C;

//     // 在设备上分配全局内存
//     CUDA_CHECK(cudaMalloc(&d_A, size));
//     CUDA_CHECK(cudaMalloc(&d_B, size));
//     CUDA_CHECK(cudaMalloc(&d_C, size));

//     // 将数据从主机拷贝到设备
//     CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

//     // 配置核函数启动参数
//     int threadsPerBlock = 256;
//     // 计算需要的线程块数量，确保能覆盖所有N个元素
//     int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

//     // 启动核函数
//     vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
//     // 检查核函数启动是否有错误
//     CUDA_CHECK(cudaGetLastError());
//     CUDA_CHECK(cudaDeviceSynchronize()); // 等待GPU完成所有任务

//     // 将结果从设备拷贝回主机
//     CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

//     // 验证结果
//     bool success = true;
//     for (int i = 0; i < 10; ++i) { // 只验证前10个
//         if (abs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
//             success = false;
//             break;
//         }
//     }
//     if (success) {
//         std::cout << "Vector addition successful!" << std::endl;
//         std::cout << "A[0] = " << h_A[0] << ", B[0] = " << h_B[0] << ", C[0] = " << h_C[0] << std::endl;
//     } else {
//         std::cout << "Vector addition failed!" << std::endl;
//     }

//     // 释放内存
//     delete[] h_A;
//     delete[] h_B;
//     delete[] h_C;
//     CUDA_CHECK(cudaFree(d_A));
//     CUDA_CHECK(cudaFree(d_B));
//     CUDA_CHECK(cudaFree(d_C));

//     return 0;
// }



#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ==================== CUDA Kernels ====================
__global__ void thomasAlgorithmKernelGlobal(
    const double* lower, const double* diag, const double* upper,
    double* inverse, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col >= n) return;
    
    double* L_lower = new double[n-1];
    double* U_diag = new double[n];
    double* U_upper = new double[n-1];
    double* y = new double[n];
    
    // LU decomposition
    U_diag[0] = diag[0];
    if (n > 1) U_upper[0] = upper[0];
    
    for (int i = 1; i < n; i++) {
        L_lower[i-1] = lower[i-1] / U_diag[i-1];
        U_diag[i] = diag[i] - L_lower[i-1] * upper[i-1];
        if (i < n - 1) U_upper[i] = upper[i];
    }
    
    // Forward substitution
    y[0] = (col == 0) ? 1.0 : 0.0;
    for (int i = 1; i < n; i++) {
        double b_i = (col == i) ? 1.0 : 0.0;
        y[i] = b_i - L_lower[i-1] * y[i-1];
    }
    
    // Backward substitution
    double x_val = y[n-1] / U_diag[n-1];
    inverse[(n-1) * n + col] = x_val;
    
    for (int i = n - 2; i >= 0; i--) {
        x_val = (y[i] - U_upper[i] * x_val) / U_diag[i];
        inverse[i * n + col] = x_val;
    }
    
    delete[] L_lower;
    delete[] U_diag;
    delete[] U_upper;
    delete[] y;
}

__global__ void matrixMultiplyKernel(
    const double* A, const double* B, double* C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// ==================== Tridiagonal Matrix Class ====================
class TridiagonalMatrix {
public:
    int n;
    vector<double> lower;
    vector<double> diag;
    vector<double> upper;

    TridiagonalMatrix(int size) : n(size) {
        lower.resize(n - 1);
        diag.resize(n);
        upper.resize(n - 1);
    }

    void generateInvertible() {
        srand(time(0));
        
        for (int i = 0; i < n - 1; i++) {
            lower[i] = (rand() % 20 - 10) + (rand() % 100) / 100.0;
            upper[i] = (rand() % 20 - 10) + (rand() % 100) / 100.0;
            
            if (fabs(lower[i]) < 0.1) lower[i] = 1.0;
            if (fabs(upper[i]) < 0.1) upper[i] = 1.0;
        }
        
        for (int i = 0; i < n; i++) {
            double sum = 0;
            if (i > 0) sum += fabs(lower[i - 1]);
            if (i < n - 1) sum += fabs(upper[i]);
            
            diag[i] = sum + (rand() % 10 + 1);
            if (rand() % 2 == 0) diag[i] = -diag[i];
        }
    }

    void print() const {
        if (n > 10) {
            cout << "Matrix too large to display (size: " << n << "x" << n << ")" << endl;
            return;
        }
        cout << "Tridiagonal Matrix (" << n << "x" << n << "):" << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    printf("%8.2f ", diag[i]);
                } else if (i == j + 1 && i > 0) {
                    printf("%8.2f ", lower[j]);
                } else if (i == j - 1 && i < n - 1) {
                    printf("%8.2f ", upper[i]);
                } else {
                    printf("%8.2f ", 0.0);
                }
            }
            cout << endl;
        }
    }

    double determinant() const {
        if (n == 1) return diag[0];
        
        vector<double> d(n);
        d[0] = diag[0];
        d[1] = diag[0] * diag[1] - lower[0] * upper[0];
        
        for (int i = 2; i < n; i++) {
            d[i] = diag[i] * d[i - 1] - lower[i - 1] * upper[i - 1] * d[i - 2];
        }
        
        return d[n - 1];
    }

    vector<vector<double>> getFullMatrix() const {
        vector<vector<double>> matrix(n, vector<double>(n, 0));
        for (int i = 0; i < n; i++) {
            matrix[i][i] = diag[i];
            if (i > 0) matrix[i][i - 1] = lower[i - 1];
            if (i < n - 1) matrix[i][i + 1] = upper[i];
        }
        return matrix;
    }
};

// ==================== CPU Version ====================
class InversesTridiagonalMatrixCPU {
private:
    int n;
    vector<vector<double>> inverse;
    
    vector<double> thomasAlgorithm(const TridiagonalMatrix& matrix, const vector<double>& b) {
        int n = matrix.n;
        
        // LU decomposition
        vector<double> L_lower(n-1, 0);
        vector<double> U_diag(n, 0);
        vector<double> U_upper(n-1, 0);
        
        U_diag[0] = matrix.diag[0];
        U_upper[0] = matrix.upper[0];
        
        for (int i = 1; i < n; i++) {
            L_lower[i-1] = matrix.lower[i-1] / U_diag[i-1];
            U_diag[i] = matrix.diag[i] - L_lower[i-1] * matrix.upper[i-1];
            if (i < n-1) U_upper[i] = matrix.upper[i];
        }
        
        // Forward substitution: Ly = b
        vector<double> y(n, 0);
        y[0] = b[0];
        for (int i = 1; i < n; i++) 
            y[i] = b[i] - L_lower[i-1] * y[i-1];
        
        // Backward substitution: Ux = y
        vector<double> x(n, 0);
        x[n-1] = y[n-1] / U_diag[n-1];
        for (int i = n-2; i >= 0; i--)
            x[i] = (y[i] - U_upper[i] * x[i+1]) / U_diag[i];
        
        return x;
    }

public:
    InversesTridiagonalMatrixCPU(const TridiagonalMatrix& matrix) {
        n = matrix.n;
        inverse.resize(n, vector<double>(n, 0));
        computeInverse(matrix);
    }

    void computeInverse(const TridiagonalMatrix& matrix) {
        for (int i = 0; i < n; i++) {
            vector<double> e(n, 0);
            e[i] = 1;
            vector<double> xi = thomasAlgorithm(matrix, e);
            for (int j = 0; j < n; j++)
                inverse[j][i] = xi[j];
        }
    }

    vector<vector<double>> getInverseMatrix() const {
        return inverse;
    }
};

// ==================== GPU Version ====================
class InversesTridiagonalMatrixGPU {
private:
    int n;
    vector<vector<double>> inverse;

public:
    InversesTridiagonalMatrixGPU(const TridiagonalMatrix& matrix) {
        n = matrix.n;
        inverse.resize(n, vector<double>(n, 0));
        computeInverseGPU(matrix);
    }

    void computeInverseGPU(const TridiagonalMatrix& matrix) {
        double *d_lower, *d_diag, *d_upper, *d_inverse;
        
        CUDA_CHECK(cudaMalloc(&d_lower, (n - 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_diag, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_upper, (n - 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_inverse, n * n * sizeof(double)));
        
        CUDA_CHECK(cudaMemcpy(d_lower, matrix.lower.data(), 
                             (n - 1) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_diag, matrix.diag.data(), 
                             n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_upper, matrix.upper.data(), 
                             (n - 1) * sizeof(double), cudaMemcpyHostToDevice));
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        
        thomasAlgorithmKernelGlobal<<<blocksPerGrid, threadsPerBlock>>>(
            d_lower, d_diag, d_upper, d_inverse, n);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        vector<double> h_inverse(n * n);
        CUDA_CHECK(cudaMemcpy(h_inverse.data(), d_inverse, 
                             n * n * sizeof(double), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = h_inverse[i * n + j];
            }
        }
        
        cudaFree(d_lower);
        cudaFree(d_diag);
        cudaFree(d_upper);
        cudaFree(d_inverse);
    }

    vector<vector<double>> getInverseMatrix() const {
        return inverse;
    }
};

// ==================== Verification Functions ====================
void verifyInverseGPU(const vector<vector<double>>& A, 
                      const vector<vector<double>>& B,
                      vector<vector<double>>& C, int n) {
    double *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    vector<double> h_A(n * n), h_B(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A[i * n + j] = A[i][j];
            h_B[i * n + j] = B[i][j];
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + 15) / 16, (n + 15) / 16);
    
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    vector<double> h_C(n * n);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = h_C[i * n + j];
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

double calculateError(const vector<vector<double>>& A, const vector<vector<double>>& B, int n) {
    double maxError = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double diff = fabs(A[i][j] - B[i][j]);
            if (diff > maxError) maxError = diff;
        }
    }
    return maxError;
}

void printMatrix(const vector<vector<double>>& matrix, const string& name) {
    int n = matrix.size();
    if (n > 10) {
        cout << name << " is too large to display (size: " << n << "x" << n << ")" << endl;
        return;
    }
    cout << name << ":" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.4f ", matrix[i][j]);
        }
        cout << endl;
    }
}

// ==================== Performance Test ====================
void performanceTest(int n) {
    cout << "\n========================================" << endl;
    cout << "Performance Test for Matrix Size: " << n << "x" << n << endl;
    cout << "========================================" << endl;
    
    TridiagonalMatrix matrix(n);
    matrix.generateInvertible();
    
    double det = matrix.determinant();
    cout << "Determinant: " << det << endl;
    
    if (fabs(det) < 1e-10) {
        cout << "Matrix is not invertible!" << endl;
        return;
    }
    
    // CPU Version
    cout << "\n--- CPU Version ---" << endl;
    auto cpu_start = high_resolution_clock::now();
    InversesTridiagonalMatrixCPU inverseCPU(matrix);
    auto cpu_end = high_resolution_clock::now();
    auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
    vector<vector<double>> inverse_cpu = inverseCPU.getInverseMatrix();
    cout << "CPU Time: " << cpu_duration.count() / 1000.0 << " ms" << endl;
    
    // GPU Version
    cout << "\n--- GPU Version ---" << endl;
    auto gpu_start = high_resolution_clock::now();
    InversesTridiagonalMatrixGPU inverseGPU(matrix);
    auto gpu_end = high_resolution_clock::now();
    auto gpu_duration = duration_cast<microseconds>(gpu_end - gpu_start);
    vector<vector<double>> inverse_gpu = inverseGPU.getInverseMatrix();
    cout << "GPU Time: " << gpu_duration.count() / 1000.0 << " ms" << endl;
    
    // Calculate Speedup
    double speedup = (double)cpu_duration.count() / gpu_duration.count();
    cout << "\n--- Performance Summary ---" << endl;
    cout << "Speedup (CPU/GPU): " << speedup << "x" << endl;
    
    if (speedup > 1.0) {
        cout << "GPU is " << speedup << "x FASTER than CPU" << endl;
    } else {
        cout << "CPU is " << (1.0/speedup) << "x FASTER than GPU" << endl;
    }
    
    // Verify correctness
    double error = calculateError(inverse_cpu, inverse_gpu, n);
    cout << "\nMax difference between CPU and GPU results: " << scientific << error << endl;
    
    // Print matrices for small sizes
    if (n <= 10) {
        matrix.print();
        cout << endl;
        printMatrix(inverse_cpu, "CPU Inverse Matrix");
        cout << endl;
        printMatrix(inverse_gpu, "GPU Inverse Matrix");
    }
    
    // Verify A * A^-1 = I
    if (n <= 1000) {
        vector<vector<double>> identity(n, vector<double>(n, 0));
        vector<vector<double>> full_matrix = matrix.getFullMatrix();
        verifyInverseGPU(full_matrix, inverse_gpu, identity, n);
        
        double id_error = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double diff = fabs(identity[i][j] - expected);
                if (diff > id_error) id_error = diff;
            }
        }
        cout << "Verification Error (A * A^-1 - I): " << scientific << id_error << endl;
        
        if (n <= 10) {
            cout << endl;
            printMatrix(identity, "Verification A * A^-1");
        }
    }
}

// ==================== Main ====================
int main() {
    cout << "==================================================" << endl;
    cout << "  Tridiagonal Matrix Inversion: CPU vs GPU  " << endl;
    cout << "==================================================" << endl;
    
    // Test different matrix sizes
    vector<int> sizes = {10, 50, 100, 500, 1000, 2000, 5000};
    
    cout << "\nChoose test mode:" << endl;
    cout << "1. Single size test" << endl;
    cout << "2. Multiple sizes benchmark" << endl;
    cout << "Enter choice (1 or 2): ";
    
    int choice;
    cin >> choice;
    
    if (choice == 1) {
        int n;
        cout << "Enter matrix dimension: ";
        cin >> n;
        performanceTest(n);
    } else {
        cout << "\n==================================================" << endl;
        cout << "  Running Benchmark for Multiple Matrix Sizes  " << endl;
        cout << "==================================================" << endl;
        
        cout << "\nSummary Table:" << endl;
        cout << "-------------------------------------------------------------------" << endl;
        printf("%-10s %-15s %-15s %-15s\n", "Size", "CPU Time(ms)", "GPU Time(ms)", "Speedup");
        cout << "-------------------------------------------------------------------" << endl;
        
        for (int n : sizes) {
            TridiagonalMatrix matrix(n);
            matrix.generateInvertible();
            
            // CPU
            auto cpu_start = high_resolution_clock::now();
            InversesTridiagonalMatrixCPU inverseCPU(matrix);
            auto cpu_end = high_resolution_clock::now();
            auto cpu_time = duration_cast<microseconds>(cpu_end - cpu_start).count() / 1000.0;
            
            // GPU
            auto gpu_start = high_resolution_clock::now();
            InversesTridiagonalMatrixGPU inverseGPU(matrix);
            auto gpu_end = high_resolution_clock::now();
            auto gpu_time = duration_cast<microseconds>(gpu_end - gpu_start).count() / 1000.0;
            
            double speedup = cpu_time / gpu_time;
            
            printf("%-10d %-15.3f %-15.3f %-15.3fx\n", n, cpu_time, gpu_time, speedup);
        }
        cout << "-------------------------------------------------------------------" << endl;
    }
    
    return 0;
}