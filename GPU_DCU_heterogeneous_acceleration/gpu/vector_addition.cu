#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

//cuda错误检查宏，gpu调试核心//copy//failure
//do...while(0),确保文本替换时，替换的是一整坨。
//cudaError_t,cuda错误类型枚举
#define CHECK(call)\
    do{\
        cudaError_t err=call;\
        if(err!=cudaSuccess){\
            printf("cuda错误%s行号：%d信息：%s\n",__FILE__,__LINE__,cudaGetErrorString(err));\
            exit(EXIT_FAILURE);\
        }\
    }while(0)

//cuda核函数，gpu上执行的核心代码。
__global__ void vectorAddKernel(const float* a,const float* b,float* c,int n){//const与*搭配！global一边是两个下滑横线！
    //计算当前线程的全局索引（核心逻辑）
    int idx=blockIdx.x*blockDim.x+threadIdx.x;//数万线程同时执行这行代码
    //blockIdx.x:线程块block在线程格grid中的索引
    //blockDim.x:每个线程块block中线程数量
    //threadIdx.x:线程块block中线程索引
    //每个线程对应一个唯一的idx，负责计算出c[idx]

    //线程数在分配的过程中会多分配的哦
    //防越界
    if(idx<n) c[idx]=a[idx]+b[idx];//每个线程处理一个向量元素的加法
}//只能由cpu调用，在gpu的sm（流多处理器）上并行执行

//cpu端：操作全流程，封装函数
void vectorAddCUDA(const float *h_a,const float *h_b,float *h_c,int n){//h_c是可改动的引用
    //step 1:gpu分配显存
    float *d_a,*d_b,*d_c;//指向gpu显存，cpu不读写
    CHECK(cudaMalloc(&d_a,n*sizeof(float)));//为指针换一个指向
    CHECK(cudaMalloc(&d_b,n*sizeof(float)));
    CHECK(cudaMalloc(&d_c,n*sizeof(float)));
    //step 2:copy data cpu->gpu
    CHECK(cudaMemcpy(d_a,h_a,n*sizeof(float),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b,h_b,n*sizeof(float),cudaMemcpyHostToDevice));
    //step 3:start kernel(gpu并行计算)
    int blockSize=256;//线程块的线程数，匹配gpu warp大小
    int gridSize=(n+blockSize-1)/blockSize;//向上取整，所有元素能够被线程覆盖
    //start kernel语法
    vectorAddKernel<<<gridSize,blockSize>>>(d_a,d_b,d_c,n);
    CHECK(cudaGetLastError());//start kernel错误无法直接捕获
    CHECK(cudaDeviceSynchronize());//等待gpu计算完成!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //step 4:copy result gpu->cpu
    CHECK(cudaMemcpy(h_c,d_c,n*sizeof(float),cudaMemcpyDeviceToHost));
    //step 5:free gpu memory
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
}

int main(){
    //step 1:initial cpu data
    const int n=10000;
    float *h_a=new float[n];//动态分配堆内存
    float *h_b=new float[n];
    float *h_c=new float[n];
    float *h_ref=new float[n];//存储cpu计算的参考结果用于验证
    for(int i=0;i<n;i++){
        h_a[i]=static_cast<float>(rand())/RAND_MAX;
        h_b[i]=static_cast<float>(rand())/RAND_MAX;
        h_ref[i]=h_a[i]+h_b[i];//纯cpu计算结果
    }
    //step 2:call vectorAddCUDA
    vectorAddCUDA(h_a,h_b,h_c,n);
    //step 3:verify result，要考虑浮点数误差
    bool isCorrect=true;
    for(int i=0;i<n;i++){
        if(fabs(h_ref[i]-h_c[i])>1e-5){
            isCorrect=false;
            printf("failure\n");
            break;
        }
    }
    if(isCorrect) printf("success\n");
    //step 4:free cpu memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_ref;
    //reset device，释放gpu资源
    CHECK(cudaDeviceReset());
    return 0;
}