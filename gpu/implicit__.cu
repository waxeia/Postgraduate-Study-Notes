#include <cuda_runtime.h>
#include <cstdio>//cuda场景比iostream更合适
#include <cmath>
#include <cstdlib>//exit
#include <vector>
#include <algorithm>//find
#include <string>

using namespace std;

//宏定义后面不可加分号
#define CUDART_PI_F 3.141592654f//单精度pi
#define CUDART_PI 3.14159265358979323846//双精度pi

//cuda错误检查宏（宏是预编译替换，cpu和gpu都能用）fprintf 支持 const char* 参数
#define CUDA_CHECK(call)\
    do{\
        cudaError_t err=call;\
        if(err!=cudaSuccess){\
            fprintf(stderr,"CUDA error:%s;%d;%s\n",__FILE__,__LINE__,cudaGetErrorString(err));\
            exit(1);\
        }\
    }while(0)

void thomas(const vector<double>& lower,const vector<double>& diag,const vector<double>& upper,const vector<double>& b,vector<double>& x){
    int n=diag.size();
    //LU分解
    vector<double> u_diag(n,0.0);
    vector<double> l_lower(n,0.0);
    vector<double> u_upper(n,0.0);
//追
    //initial
    u_diag[0]=diag[0];
    u_upper[0]=upper[0];
        for (int i = 1; i < n; i++) {
        l_lower[i-1] = lower[i-1] / u_diag[i-1];
        u_diag[i] = diag[i] - l_lower[i-1] * u_upper[i-1];
        if (i < n-1) u_upper[i] = upper[i];
    }//LU--done
    //forward substitution:Ly=b
    vector<double> y(n, 0.0);
    y[0] = b[0];//initial
    for(int i=1;i<n;i++) y[i]=b[i]-l_lower[i-1]*y[i-1];
//赶
    //backward substitution:Ux=y
    x[n-1]=y[n-1]/u_diag[n-1];//initial
    for(int i=n-2;i>=0;i--) x[i]=(y[i]-u_upper[i]*x[i+1])/u_diag[i];
}

//计算右端项K的核函数
__global__ void compute_K_kernel(const double *u,double *K,double E,int N){
    //计算线程的全局索引，并且计算的是内点，K从1到N-1
    int j=blockIdx.x*blockDim.x+threadIdx.x+1;
    //边界检查：启动的线程数通常向上取整（比如需要 100 线程，启动 128），多余线程必须跳过，否则会访问非法内存导致崩溃。
    if(j<N) K[j-1]=(1.0-E)*u[j]+0.5*E*(u[j+1]+u[j-1]);//K从0到N-2
}

//更新速度的核函数
__global__ void update_u_kernel(const double *x,double *u,int N){
    int j=blockIdx.x*blockDim.x+threadIdx.x+1;
    if(j<N) u[j]=x[j-1];//x从0到N-2,u从1到N-1,u[0]和u[N]分别是上下边界值
}//一个线程处理一个数据元素的并行模式

//计算精确解的核函数----waitting for understanding......
__global__ void couette_exact_kernel(const double *y,double *u_exact,double t_star,double ReD,int n_points,int n_terms){
    int i=blockIdx.x*blockDim.x+threadIdx.x;//y数组索引从0开始
    if(i<n_points){
        double base=y[i];
        double s=0.0;
        const double pi=CUDART_PI;  // 设备端使用PI宏（预编译替换）
        // 每个线程独立计算求和项（并行化关键：线程间无数据依赖）
        for(int m=1;m<=n_terms;m++){
            double sign=(m%2==1)?-1.0:1.0;
            s+=(sign/m)*sin(m*pi*y[i])*exp(-(m*m)*(pi*pi)*t_star/ReD);
        }
        u_exact[i]=base+(2.0/pi)*s;//每个线程输出一个精确解   
    }
}

//CouetteSolverGPU类,封装GPU内存、参数和计算流程
class CouetteSolverGPU{
private:
    int N;
    int dimension;
    double ReD;
    double E;
    double dy;
    double dt;
    double A,B;
    //host cpu内存
    vector<double> h_u;
    vector<double> h_K;
    vector<double> h_x;
    vector<double> lower,diag,upper;
    //存储device gpu内存地址,实质是host指针
    double *d_u;
    double *d_K;
    double *d_x;
//构造函数
public:
    CouetteSolverGPU(double ReD_,int N_,double E_):ReD(ReD_),N(N_),E(E_){
        dimension=N-1;
        dy=1.0/N;
        dt=E*ReD*dy*dy;
        A=-E/2.0;
        B=1.0+E;
        //分配cpu内存--initial
        h_u.resize(N+1,0.0);
        h_u[0]=0.0;//下边界
        h_u[N]=1.0;//上边界
        h_K.resize(dimension);
        h_x.resize(dimension);
        lower.resize(dimension-1,A);
        diag.resize(dimension,B);
        upper.resize(dimension-1,A);
        //分配gpu内存
        CUDA_CHECK(cudaMalloc(&d_u,(N+1)*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_K,dimension*sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x,dimension*sizeof(double)));
        //将初始速度h_u从cpu拷贝到gpu
        CUDA_CHECK(cudaMemcpy(d_u,h_u.data(),(N+1)*sizeof(double),cudaMemcpyHostToDevice));
        //析构函数（释放GPU内存）
    }
    ~CouetteSolverGPU(){
        // 释放GPU内存（避免显存泄漏）
        cudaFree(d_u);
        cudaFree(d_K);
        cudaFree(d_x);
    }
    void step(){
        //配置核函数启动参数
        int blockSize=256;
        int numBlocks=(dimension+blockSize-1)/blockSize;
        //启动核函数算K
        compute_K_kernel<<<numBlocks,blockSize>>>(d_u,d_K,E,N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());//CPU等待GPU执行完
        CUDA_CHECK(cudaMemcpy(h_K.data(),d_K,dimension*sizeof(double),cudaMemcpyDeviceToHost));//将K从GPU拷贝回CPU
        //在CPU上执行thomas算法
        h_K[dimension-1]-=A*h_u[N];//修改右端项b
        thomas(lower,diag,upper,h_K,h_x);
        //将解x从CPU拷贝到GPU
        CUDA_CHECK(cudaMemcpy(d_x,h_x.data(),dimension*sizeof(double),cudaMemcpyHostToDevice));
        update_u_kernel<<<numBlocks,blockSize>>>(d_x,d_u,N);//启动核函数更新速度u
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        //更新后的d_u拷贝回CPU的h_u
        CUDA_CHECK(cudaMemcpy(h_u.data(),d_u,(N+1)*sizeof(double),cudaMemcpyDeviceToHost));
    }
    //获取解/参数
    void get_solution(std::vector<double>& u) {
        u.resize(N + 1);
        // 将GPU的d_u拷贝到CPU的u中（D2H）
        CUDA_CHECK(cudaMemcpy(u.data(),d_u,(N + 1)*sizeof(double),cudaMemcpyDeviceToHost));
    }
    double get_dt() const { return dt; }
    int get_N() const { return N; }
};

void couette_exact(const std::vector<double>& y, double t_star, double ReD,
                   std::vector<double>& u_exact, int n_terms = 800) {
    int n_points = y.size();
    u_exact.resize(n_points);
    
    // 分配临时GPU内存
    double *d_y, *d_u_exact;
    CUDA_CHECK(cudaMalloc(&d_y, n_points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_exact, n_points * sizeof(double)));
    
    // H2D拷贝：y→GPU
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), n_points * sizeof(double), 
                         cudaMemcpyHostToDevice));
    
    // 启动核函数
    int blockSize = 256;
    int numBlocks = (n_points + blockSize - 1) / blockSize;
    couette_exact_kernel<<<numBlocks, blockSize>>>(d_y, d_u_exact, t_star,ReD, n_points, n_terms);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); 
    // D2H拷贝：精确解→CPU
    CUDA_CHECK(cudaMemcpy(u_exact.data(), d_u_exact, n_points * sizeof(double), 
                         cudaMemcpyDeviceToHost));
    // 释放临时GPU内存
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
    // 创建GPU求解器（初始化GPU内存和参数）
    CouetteSolverGPU solver(ReD, N, E);
    // 初始化y坐标（CPU）
    std::vector<double> y(N + 1);
    for (int i = 0; i <= N; i++) {
        y[i] = static_cast<double>(i) / N;
    }
    // 输出参数
    std::printf("Dy* = %.6f,  Dt* = %.6f  (E=%.1f, ReD=%.1f)\n", 
                1.0/N, solver.get_dt(), E, ReD);
    std::vector<double> u_num, u_exact;
    // 迭代求解
    for (int n = 0; n <= steps_to_save.back(); n++) {
        // 保存指定步数的结果
        if (std::find(steps_to_save.begin(), steps_to_save.end(), n) != steps_to_save.end()) {
            solver.get_solution(u_num);  // 从GPU拷贝数值解到CPU
            double t_star = n * solver.get_dt();
            if (show_ana) {
                couette_exact(y, t_star, ReD, u_exact, 800);  // 调用GPU计算精确解
            } 
            // 输出结果（CPU端）
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
        // 执行一步求解（调用GPU计算）
        if (n < steps_to_save.back()) {
            solver.step();
        }
    }
    return 0;
}