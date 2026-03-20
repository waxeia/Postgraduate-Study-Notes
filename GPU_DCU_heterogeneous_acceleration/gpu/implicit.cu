//托马斯算法没有放到gpu上

#include <cuda_runtime.h>
#include <cstdio>    // 替代 stdio.h
#include <cmath>     // 替代 math.h
#include <cstdlib>   // 替代 stdlib.h
#include <vector>
#include <algorithm>
#include <string>

const double M_PI = 3.14159265358979323846;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

// Thomas算法CPU实现
void thomas_solve(const double* a, const double* b, const double* c, 
                  const double* d, double* x, int n) {
    std::vector<double> cp(n);
    std::vector<double> dp(n);
    
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    
    for (int i = 1; i < n - 1; i++) {
        double denom = b[i] - a[i-1] * cp[i-1];
        cp[i] = c[i] / denom;
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom;
    }
    
    double denom = b[n-1] - a[n-2] * cp[n-2];
    dp[n-1] = (d[n-1] - a[n-2] * dp[n-2]) / denom;
    
    x[n-1] = dp[n-1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = dp[i] - cp[i] * x[i+1];
    }
}

__global__ void compute_rhs_kernel(const double* u, double* K, 
                                   double E, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (j < N) {
        K[j-1] = (1.0 - E) * u[j] + 0.5 * E * (u[j+1] + u[j-1]);
    }
}

class CouetteSolverGPU {
private:
    int N;
    int n_int;
    double ReD;
    double E;
    double dy;
    double dt;
    double A_val, B_val;
    
    double *d_u;
    double *d_K;
    
    std::vector<double> h_a, h_b, h_c;
    std::vector<double> h_K;
    std::vector<double> h_u_int;
    std::vector<double> h_u;
    
public:
    CouetteSolverGPU(double ReD_, int N_, double E_) 
        : ReD(ReD_), N(N_), E(E_) {
        
        n_int = N - 1;
        dy = 1.0 / N;
        dt = E * ReD * (dy * dy);
        
        A_val = -E / 2.0;
        B_val = 1.0 + E;
        
        h_a.assign(n_int - 1, A_val);
        h_b.assign(n_int, B_val);
        h_c.assign(n_int - 1, A_val);
        h_K.resize(n_int);
        h_u_int.resize(n_int);
        h_u.assign(N + 1, 0.0);
        h_u[N] = 1.0;
        
        CUDA_CHECK(cudaMalloc(&d_u, (N + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_K, n_int * sizeof(double)));
        
        CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), (N + 1) * sizeof(double), 
                             cudaMemcpyHostToDevice));
    }
    
    ~CouetteSolverGPU() {
        cudaFree(d_u);
        cudaFree(d_K);
    }
    
    void step() {
        int blockSize = 256;
        int numBlocks = (n_int + blockSize - 1) / blockSize;
        compute_rhs_kernel<<<numBlocks, blockSize>>>(d_u, d_K, E, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(h_K.data(), d_K, n_int * sizeof(double), 
                             cudaMemcpyDeviceToHost));
        
        h_K[n_int - 1] -= A_val * 1.0;
        
        thomas_solve(h_a.data(), h_b.data(), h_c.data(), 
                    h_K.data(), h_u_int.data(), n_int);
        
        h_u[0] = 0.0;
        for (int i = 0; i < n_int; i++) {
            h_u[i + 1] = h_u_int[i];
        }
        h_u[N] = 1.0;
        
        CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), (N + 1) * sizeof(double), 
                             cudaMemcpyHostToDevice));
    }
    
    void get_solution(std::vector<double>& u) {
        u.resize(N + 1);
        CUDA_CHECK(cudaMemcpy(u.data(), d_u, (N + 1) * sizeof(double), 
                             cudaMemcpyDeviceToHost));
    }
    
    double get_dt() const { return dt; }
};

void couette_analytic(const std::vector<double>& y, double t_star, double ReD,
                     std::vector<double>& u_ana, int n_terms = 800) {
    u_ana.resize(y.size());
    const double pi = M_PI;
    
    for (size_t i = 0; i < y.size(); i++) {
        double base = y[i];
        double s = 0.0;
        
        for (int m = 1; m <= n_terms; m++) {
            double sign = (m % 2 == 1) ? -1.0 : 1.0;
            s += (sign / m) * std::sin(m * pi * y[i]) * 
                 std::exp(-(m * m) * (pi * pi) * t_star / ReD);
        }
        
        u_ana[i] = base + (2.0 / pi) * s;
    }
}

std::string book_sci(double x, int digits = 3) {
    char buffer[64];
    
    if (std::fabs(x) < 1e-99) {
        std::sprintf(buffer, ".%0*dE+00", digits, 0);
        return std::string(buffer);
    }
    
    double ax = std::fabs(x);
    std::sprintf(buffer, "%.*e", digits, ax);
    
    std::string s(buffer);
    size_t e_pos = s.find('e');
    if (e_pos == std::string::npos) e_pos = s.find('E');
    
    double m = std::atof(s.substr(0, e_pos).c_str());
    int e = std::atoi(s.substr(e_pos + 1).c_str());
    
    m = m / 10.0;
    e = e + 1;
    
    double factor = std::pow(10.0, digits);
    m = std::round(m * factor) / factor;
    
    if (m >= 1.0) {
        m /= 10.0;
        e += 1;
    }
    
    int frac = static_cast<int>(std::round(m * factor));
    int abs_e = (e < 0) ? -e : e;
    std::sprintf(buffer, "%s.%0*dE%c%02d", 
            (x < 0) ? "-" : "",
            digits, frac,
            (e >= 0) ? '+' : '-',
            abs_e);
    
    return std::string(buffer);
}

int main() {
    double ReD = 5000.0;
    int N = 20;
    double E = 1.0;
    std::vector<int> output_steps = {12, 36, 60, 120, 240, 360};
    
    CouetteSolverGPU solver(ReD, N, E);
    
    std::vector<double> y(N + 1);
    for (int i = 0; i <= N; i++) {
        y[i] = static_cast<double>(i) / N;
    }
    
    std::printf("Δy* = %.6f,  Δt* = %.6f  (E=%.1f, ReD=%.1f)\n", 
           1.0/N, solver.get_dt(), E, ReD);
    
    std::vector<double> u_num, u_ana;
    
    for (int n = 0; n <= output_steps.back(); n++) {
        if (std::find(output_steps.begin(), output_steps.end(), n) != output_steps.end()) {
            solver.get_solution(u_num);
            double t_star = n * solver.get_dt();
            
            couette_analytic(y, t_star, ReD, u_ana, 800);
            
            std::printf("\n========================================================================\n");
            std::printf("n = %4d steps,  t* = n·Δt* = %.6f\n", n, t_star);
            std::printf(" j   y/D      u/u_e (num)   u/u_e (ana)\n");
            
            for (int j = 0; j <= N; j++) {
                std::printf("%2d  %.2f     %9s     %9s\n", 
                       j, y[j], 
                       book_sci(u_num[j], 3).c_str(), 
                       book_sci(u_ana[j], 3).c_str());
            }
        }
        
        if (n < output_steps.back()) {
            solver.step();
        }
    }
    
    return 0;
}
