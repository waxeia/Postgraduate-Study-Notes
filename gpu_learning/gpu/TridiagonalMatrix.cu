#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

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
    
    U_diag[0] = diag[0];
    if (n > 1) U_upper[0] = upper[0];
    
    for (int i = 1; i < n; i++) {
        L_lower[i-1] = lower[i-1] / U_diag[i-1];
        U_diag[i] = diag[i] - L_lower[i-1] * upper[i-1];
        if (i < n - 1) U_upper[i] = upper[i];
    }
    
    y[0] = (col == 0) ? 1.0 : 0.0;
    for (int i = 1; i < n; i++) {
        double b_i = (col == i) ? 1.0 : 0.0;
        y[i] = b_i - L_lower[i-1] * y[i-1];
    }
    
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

class InversesTridiagonalMatrixCUDA {
private:
    int n;
    vector<vector<double>> inverse;

public:
    InversesTridiagonalMatrixCUDA(const TridiagonalMatrix& matrix) {
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

int main() {
    int n;
    cout << "Enter matrix dimension: ";
    cin >> n;

    TridiagonalMatrix matrix(n);
    matrix.generateInvertible();
    
    cout << endl;
    matrix.print();
    
    cout << "\nDeterminant: " << matrix.determinant() << endl;
    
    if (fabs(matrix.determinant()) > 1e-10) {
        cout << "Matrix is invertible!" << endl;
        
        InversesTridiagonalMatrixCUDA inverseMatrix(matrix);
        vector<vector<double>> inverse_matrix = inverseMatrix.getInverseMatrix();

        vector<vector<double>> identity(n, vector<double>(n, 0));
        vector<vector<double>> full_matrix = matrix.getFullMatrix();
        
        verifyInverseGPU(full_matrix, inverse_matrix, identity, n);
        
        cout << "\nVerification A * A^-1 = I (GPU computed):" << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%8.4f ", identity[i][j]);
            }
            cout << endl;
        }

        cout << "\nInverse Matrix:" << endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%8.4f ", inverse_matrix[i][j]);
            }
            cout << endl;
        }
    } else {
        cout << "Matrix is not invertible (determinant near 0)" << endl;
    }
    
    return 0;
}