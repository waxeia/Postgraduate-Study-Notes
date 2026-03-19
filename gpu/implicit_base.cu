#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>

// Device constant for PI - CUDA device code can access
#define CUDART_PI_F 3.141592654f
#define CUDART_PI   3.14159265358979323846

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// CPU端托马斯算法 - 保持LU分解思想
void thomas(const std::vector<double>& lower, const std::vector<double>& diag,
            const std::vector<double>& upper, const std::vector<double>& b,
            std::vector<double>& x) {
    int n = diag.size();
    
    // LU分解decompose
    std::vector<double> u_diag(n, 0.0);
    std::vector<double> l_lower(n-1, 0.0);
    std::vector<double> u_upper(n-1, 0.0);
    
    // 追
    // initial
    u_diag[0] = diag[0];
    if (n > 1) u_upper[0] = upper[0];
    
    // recurrence
    for (int i = 1; i < n; i++) {
        l_lower[i-1] = lower[i-1] / u_diag[i-1];
        u_diag[i] = diag[i] - l_lower[i-1] * u_upper[i-1];
        if (i < n-1) u_upper[i] = upper[i];
    }
    
    // LU---done
    // forward substitution Ly=b
    std::vector<double> y(n, 0.0);
    y[0] = b[0];
    for (int i = 1; i < n; i++) {
        y[i] = b[i] - l_lower[i-1] * y[i-1];
    }
    
    // 赶
    // backward substitution Ux=y
    x.resize(n);
    x[n-1] = y[n-1] / u_diag[n-1];
    for (int i = n-2; i >= 0; i--) {
        x[i] = (y[i] - u_upper[i] * x[i+1]) / u_diag[i];
    }
}

// GPU核函数：计算右端项K
__global__ void compute_K_kernel(const double* u, double* K, 
                                 double E, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (j < N) {
        K[j-1] = (1.0 - E) * u[j] + 0.5 * E * (u[j+1] + u[j-1]);
    }
}

// GPU核函数：更新内点速度
__global__ void update_u_kernel(const double* x, double* u, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (j < N) {
        u[j] = x[j-1];
    }
}

// GPU核函数：计算精确解
__global__ void couette_exact_kernel(const double* y, double* u_exact,
                                     double t_star, double ReD,
                                     int n_points, int n_terms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_points) {
        double base = y[i];
        double s = 0.0;
        const double pi = CUDART_PI;  // Use CUDA's built-in PI constant
        
        for (int m = 1; m <= n_terms; m++) {
            double sign = (m % 2 == 1) ? -1.0 : 1.0;
            s += (sign / m) * sin(m * pi * y[i]) * 
                 exp(-(m * m) * (pi * pi) * t_star / ReD);
        }
        
        u_exact[i] = base + (2.0 / pi) * s;
    }
}

class CouetteSolverGPU {
private:
    int N;
    int dimension;
    double ReD;
    double E;
    double dy;
    double dt;
    double A, B;
    
    std::vector<double> h_u;
    std::vector<double> h_K;
    std::vector<double> h_x;
    std::vector<double> lower, diag, upper;
    
    double *d_u;
    double *d_K;
    double *d_x;
    
public:
    CouetteSolverGPU(double ReD_, int N_, double E_) 
        : ReD(ReD_), N(N_), E(E_) {
        
        dimension = N - 1;
        dy = 1.0 / N;
        dt = E * ReD * (dy * dy);
        
        A = -E / 2.0;
        B = 1.0 + E;
        
        h_u.resize(N + 1, 0.0);
        h_u[0] = 0.0;
        h_u[N] = 1.0;
        
        h_K.resize(dimension);
        h_x.resize(dimension);
        
        lower.resize(dimension - 1, A);
        diag.resize(dimension, B);
        upper.resize(dimension - 1, A);
        
        CUDA_CHECK(cudaMalloc(&d_u, (N + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_K, dimension * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x, dimension * sizeof(double)));
        
        CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), (N + 1) * sizeof(double), 
                             cudaMemcpyHostToDevice));
    }
    
    ~CouetteSolverGPU() {
        cudaFree(d_u);
        cudaFree(d_K);
        cudaFree(d_x);
    }
    
    void step() {
        int blockSize = 256;
        int numBlocks = (dimension + blockSize - 1) / blockSize;
        
        compute_K_kernel<<<numBlocks, blockSize>>>(d_u, d_K, E, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_K.data(), d_K, dimension * sizeof(double), 
                             cudaMemcpyDeviceToHost));
        
        h_K[dimension - 1] -= A * h_u[N];
        
        thomas(lower, diag, upper, h_K, h_x);
        
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), dimension * sizeof(double), 
                             cudaMemcpyHostToDevice));
        
        update_u_kernel<<<numBlocks, blockSize>>>(d_x, d_u, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_u.data(), d_u, (N + 1) * sizeof(double), 
                             cudaMemcpyDeviceToHost));
    }
    
    void get_solution(std::vector<double>& u) {
        u.resize(N + 1);
        CUDA_CHECK(cudaMemcpy(u.data(), d_u, (N + 1) * sizeof(double), 
                             cudaMemcpyDeviceToHost));
    }
    
    double get_dt() const { return dt; }
    int get_N() const { return N; }
};

void couette_exact(const std::vector<double>& y, double t_star, double ReD,
                   std::vector<double>& u_exact, int n_terms = 800) {
    int n_points = y.size();
    u_exact.resize(n_points);
    
    double *d_y, *d_u_exact;
    CUDA_CHECK(cudaMalloc(&d_y, n_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_exact, n_points * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), n_points * sizeof(double), 
                         cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int numBlocks = (n_points + blockSize - 1) / blockSize;
    couette_exact_kernel<<<numBlocks, blockSize>>>(d_y, d_u_exact, t_star, 
                                                    ReD, n_points, n_terms);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(u_exact.data(), d_u_exact, n_points * sizeof(double), 
                         cudaMemcpyDeviceToHost));
    
    cudaFree(d_y);
    cudaFree(d_u_exact);
}

std::string book_sci(double x, int digits = 3) {
    char buffer[64];
    
    if (std::fabs(x) < 1e-99) {
        std::sprintf(buffer, ".%0*dE+00", digits, 0);
        return std::string(buffer);
    }
    
    std::string sign_prefix = (x < 0) ? "-" : "";
    double abs_x = std::fabs(x);
    
    std::sprintf(buffer, "%.*E", digits, abs_x);
    std::string s(buffer);
    
    size_t e_pos = s.find('E');
    if (e_pos == std::string::npos) e_pos = s.find('e');
    
    std::string mantissa_str = s.substr(0, e_pos);
    std::string exp_str = s.substr(e_pos + 1);
    
    double m = std::atof(mantissa_str.c_str());
    int e = std::atoi(exp_str.c_str());
    
    m = m / 10.0;
    e = e + 1;
    
    double m_round = std::round(m * std::pow(10.0, digits)) / std::pow(10.0, digits);
    
    if (m_round >= 1.0) {
        m_round = m_round / 10.0;
        e = e + 1;
    }
    
    int frac = static_cast<int>(std::round(m_round * std::pow(10.0, digits)));
    std::string exp_sign = (e >= 0) ? "+" : "-";
    int exp_val = std::abs(e);
    
    std::sprintf(buffer, "%s.%0*dE%s%02d", 
                sign_prefix.c_str(), digits, frac,
                exp_sign.c_str(), exp_val);
    
    return std::string(buffer);
}

int main() {
    double ReD = 5000.0;
    int N = 20;
    double E = 1.0;
    std::vector<int> steps_to_save = {12, 36, 60, 120, 240, 360};
    bool show_ana = true;
    
    CouetteSolverGPU solver(ReD, N, E);
    
    std::vector<double> y(N + 1);
    for (int i = 0; i <= N; i++) {
        y[i] = static_cast<double>(i) / N;
    }
    
    std::printf("Dy* = %.6f,  Dt* = %.6f  (E=%.1f, ReD=%.1f)\n", 
                1.0/N, solver.get_dt(), E, ReD);
    
    std::vector<double> u_num, u_exact;
    
    for (int n = 0; n <= steps_to_save.back(); n++) {
        if (std::find(steps_to_save.begin(), steps_to_save.end(), n) != steps_to_save.end()) {
            solver.get_solution(u_num);
            double t_star = n * solver.get_dt();
            
            if (show_ana) {
                couette_exact(y, t_star, ReD, u_exact, 800);
            }
            
            std::printf("\n%s\n", std::string(72, '=').c_str());
            std::printf("n = %4d steps,  t* = n*Dt* = %.6f\n", n, t_star);
            
            if (show_ana) {
                std::printf(" j   y/D      u/u_e (numerical)   u/u_e (exact)\n");
            } else {
                std::printf(" j   y/D      u/u_e\n");
            }
            
            for (int j = 0; j <= N; j++) {
                std::string num_s = book_sci(u_num[j], 3);
                
                if (show_ana) {
                    std::string ana_s = book_sci(u_exact[j], 3);
                    std::printf("%2d  %.2f     %9s     %9s\n", 
                               j, y[j], num_s.c_str(), ana_s.c_str());
                } else {
                    std::printf("%2d  %.2f     %9s\n", 
                               j, y[j], num_s.c_str());
                }
            }
        }
        
        if (n < steps_to_save.back()) {
            solver.step();
        }
    }
    
    return 0;
}
