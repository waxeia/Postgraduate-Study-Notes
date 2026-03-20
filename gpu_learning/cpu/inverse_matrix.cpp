#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

class TridiagonalMatrix {
public:
    int n;                    // 矩阵维数
    vector<double> lower;     // 下对角线
    vector<double> diag;      // 主对角线
    vector<double> upper;     // 上对角线

    TridiagonalMatrix(int size) : n(size) {
        lower.resize(n - 1);
        diag.resize(n);
        upper.resize(n - 1);
    }

    // 生成随机可逆三对角矩阵
    void generateInvertible() {
        srand(time(0));
        
        // 随机生成上下对角线元素（非零）
        for (int i = 0; i < n - 1; i++) {
            lower[i] = (rand() % 20 - 10) + (rand() % 100) / 100.0;
            upper[i] = (rand() % 20 - 10) + (rand() % 100) / 100.0;
            
            // 确保非零
            if (fabs(lower[i]) < 0.1) lower[i] = 1.0;
            if (fabs(upper[i]) < 0.1) upper[i] = 1.0;
        }
        
        // 生成主对角线，使用对角占优保证可逆性
        for (int i = 0; i < n; i++) {
            double sum = 0;
            if (i > 0) sum += fabs(lower[i - 1]);
            if (i < n - 1) sum += fabs(upper[i]);
            
            // 主对角元素大于其他元素绝对值之和
            diag[i] = sum + (rand() % 10 + 1);
            
            // 随机正负号
            if (rand() % 2 == 0) diag[i] = -diag[i];
        }
    }

    // 打印矩阵
    void print() const {
        cout << "三对角矩阵 (" << n << "x" << n << "):" << endl;
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

    // 计算行列式（验证可逆性）
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

    // 获取完整矩阵
    vector<vector<double>> getFullMatrix() const {
        vector<vector<double>> matrix(n, vector<double>(n, 0));
        for (int i = 0; i < n; i++) {
            matrix[i][i] = diag[i];
            if (i > 0) matrix[i][i - 1] = lower[i - 1];
            if (i < n - 1) matrix[i][i + 1] = upper[i];
        }
        return matrix;
    }

    // 返回矩阵维数
    // int getSize() const { return n; }
    //const:修饰成员函数,保证函数的 “只读性”：不修改类的成员变量
};

class InversesTridiagonalMatrix{
private:
    int n;//阶数
    vector<vector<double>> inverse;//存储结果逆矩阵
    //private成员函数
    //追赶法求解Ax=b
    vector<double> thomasAlgorithm(TridiagonalMatrix& matrix,vector<double>& b){
        int n=matrix.n;
        //L&U分解
        vector<double> L_diag;
        L_diag.resize(n,1);
        vector<double> L_lower;
        L_lower.resize(n-1,0);
        vector<double> U_diag;
        U_diag.resize(n,0);
        vector<double> U_upper;
        U_upper.resize(n-1,0);
        //追
        U_diag[0]=matrix.diag[0];//initialize
        U_upper[0]=matrix.upper[0];//initialize
        for(int i=1;i<n;i++){
            L_lower[i-1]=matrix.lower[i-1]/U_diag[i-1];
            U_diag[i]=matrix.diag[i]-L_lower[i-1]*matrix.upper[i-1];
            if(i<n-1) U_upper[i]=matrix.upper[i];
        }//L&U--done
        //解Ly=b，前向替换
        vector<double> y;
        y.resize(n,0);
        y[0]=b[0];//initialize
        for(int i=1;i<n;i++) 
            y[i]=b[i]-L_lower[i-1]*y[i-1];
        //赶
        //解Ux=y，后向替换
        vector<double> x;
        x.resize(n,0);
        x[n-1]=y[n-1]/U_diag[n-1];//initialize
        for(int i=n-2;i>=0;i--)
            x[i]=(y[i]-U_upper[i]*x[i+1])/U_diag[i];
        return x;
    }

    public:
    //构造函数：对象通过构造函数来接受参数
    InversesTridiagonalMatrix(const TridiagonalMatrix& matrix){
        //const：限制引用为只读，不能修改原对象
        //&的作用：传递引用，避免拷贝，省空间
        n=matrix.n;//TODO:int
        inverse.resize(n,vector<double>(n,0));//initialize
        computeInverse(matrix);    
    }

    //public成员函数
    void computeInverse(const TridiagonalMatrix& matrix){
        //追赶法计算逆矩阵
        for(int i=0;i<n;i++){
            vector<double> e;
            e.resize(n,0);
            e[i]=1;
            // vector<double> xi=thomasAlgorithm(matrix,e);//为什么不可以直接用matrix？
            //注意原理
            vector<double> xi=thomasAlgorithm(const_cast<TridiagonalMatrix&>(matrix),e);
            for(int j=0;j<n;j++)
                inverse[j][i]=xi[j];
        }
    }

    //return inverse matrix
    vector<vector<double>> getInverseMatrix() const{
        return inverse;
    }
};

int main() {
    int n;
    cout << "请输入矩阵维数: ";
    cin >> n;

    TridiagonalMatrix matrix(n);
    matrix.generateInvertible();
    
    cout << endl;
    matrix.print();
    
    cout << "\n行列式值: " << matrix.determinant() << endl;
    
    if (fabs(matrix.determinant()) > 1e-10) {
        cout << "矩阵可逆！" << endl;
        //追赶法
        InversesTridiagonalMatrix inverseMatrix(matrix);//inverseMatrix对象名字，该句已经执行了计算逆矩阵
        vector<vector<double>> inverse_matrix=inverseMatrix.getInverseMatrix();

        //verify A*A^-1=I-------可省！
        vector<vector<double>> identity;
        identity.resize(n,vector<double>(n,0));
        vector<vector<double>> full_matrix=matrix.getFullMatrix();
        for(int i=0;i<n;i++){
            for(int j=0;j<n;j++){
                for(int k=0;k<n;k++){
                    identity[i][j]+=full_matrix[i][k]*inverse_matrix[k][j];
                }
            }
        }
        cout << "\n验证 A * A^-1 = I :" << endl;
        for (int i=0;i<n;i++) {
            for (int j=0;j<n;j++) {
                printf("%8.4f ",identity[i][j]);
            }
            cout << endl;
        }

        //打印逆矩阵
        cout << "\n逆矩阵为:" << endl;
        for (int i=0;i<n;i++) {
            for (int j=0;j<n;j++) {
                printf("%8.4f ",inverse_matrix[i][j]);
            }
            cout << endl;
        }
    } else
        cout << "矩阵不可逆（行列式接近0）" << endl; 
    return 0;
}