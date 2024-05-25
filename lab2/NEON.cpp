#include<iostream>
#include<sys/time.h>
#include<arm_neon.h>
using namespace std;
const int N=4000;
float A[N][N];
const int LOOP=1;

//初始化
void init(){
    //生成一个上三角矩阵，对角线为1.0
    for(int i=0;i<N;i++){
        for(int j=0;j<i;j++){
            A[i][j]=0.0;
        }
        A[i][i]=1.0;
        for(int k=i+1;k<N;k++){
            A[i][k]=rand()%10;
        }
    }
    //构造一个更为复杂的矩阵，但是因为只是加上行，通过初等行变化可还原成上三角矩阵
    //每一行的元素都加上其前面几行的元素
    for(int k=0;k<N;k++){
        for(int i=k+1;i<N;i++){
            for(int j=0;j<N;j++){
                A[i][j]+=A[k][j];
            }
        }
    }
}

//普通高斯消去法串行算法
void Gauss(){
    for(int k=0;k<N;k++){
        for(int j=k+1;j<N;j++){
            A[k][j]/=A[k][k];
        }
        A[k][k]=1.0;//主对角线元素变为1

        for(int i=k+1;i<N;i++){
            for(int j=k+1;j<N;j++){
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0.0;
        }
    }
}

//NEON
void Gauss_NEON(){
    for(int k=0;k<N;k++){
        float32x4_t v1 = vmovq_n_f32(A[k][k]);
        int j;
        for(j=k+1;j+4<=N;j+=4){
            //A[k][j]/=A[k][k];
            float32x4_t va=vld1q_f32(A[k] + j);
            va = vdivq_f32(va, v1);
            vst1q_f32(A[k] + j, va);
        }
        for(;j<N;j++){
            A[k][j]/=A[k][k];
        }
        A[k][k]=1.0;

        for(int i=k+1;i<N;i++){
            float32x4_t vaik = vmovq_n_f32(A[i][k]);
            for (j = k + 1; j + 4 <= N; j+=4) {
                //A[i][j] = A[i][j] - A[i][k] * A[k][j];
                float32x4_t vakj = vld1q_f32(A[k] + j);
                float32x4_t vaij = vld1q_f32(A[i] + j);
                float32x4_t vx = vmulq_f32(vaik, vakj);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(A[i] + j, vaij);
            }
            for (; j < N; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            A[i][k] = 0.0;
        }

 
    }

   
}
//NEON对齐
void Gauss_NEON_qi(){
    float32x4_t t0,t1,t2,t3;
    for(int k=0;k<N;k++)
    {
        float32x4_t v1=vld1q_dup_f32(A[k]+k);
        int j;
        for(j=k+1;j<N;j++){
               if(((size_t)(A[k]+j))%16==0)
                break;
             A[k][j]/=A[k][k];
        }
        for(;j+3<N;j+=4)
        {
            float32x4_t va=vld1q_f32(A[k]+j);
            va=vdivq_f32(va,v1);
            vst1q_f32(A[k]+j,va);
        }
        for(;j<N;j++)
            A[k][j]/=A[k][k];

        A[k][k]=1.0;
        for(int i=k+1;i<N;i++)
        {
            float32x4_t vaik=vld1q_dup_f32(A[i]+k);
            int j;
            for(j=k+1;j<N;j++){
               if(((size_t)(A[i]+j))%16==0)
                break;
             A[i][j]-=A[i][k]*A[k][j];
            }
            for(;j+4<=N;j+=4)
            {
                float32x4_t vakj=vld1q_f32(A[k]+j);
                float32x4_t vaij=vld1q_f32(A[i]+j);
                float32x4_t vx=vmulq_f32(vaik,vakj);
                vaij=vsubq_f32(vaij,vx);
                vst1q_f32(A[i]+j,vaij);
            }
            for(;j<N;j++)
                 A[i][j]-=A[i][k]*A[k][j];
            A[i][k]=0.0;
        }
    }
}

void print(){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            cout<<A[i][j]<<" ";
        }cout<<endl;
    }cout<<endl;
}


int main(){
    struct timeval start;
    struct timeval end;
    double sum_time=0.0;
    for(int i=0;i<LOOP;i++){
        init();
        gettimeofday(&start,NULL);//开始时间
        Gauss();
        gettimeofday(&end,NULL);//结束时间
        sum_time +=((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000;
    }
    cout<<"Gauss:"<<(sum_time/LOOP)<<"ms"<<endl;

    sum_time=0.0;
    for(int i=0;i<LOOP;i++){
        init();
        gettimeofday(&start,NULL);//开始时间
        Gauss_NEON();
        gettimeofday(&end,NULL);//结束时间
        sum_time +=((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000;
    }
    cout<<"Gauss_NEON:"<<(sum_time/LOOP)<<"ms"<<endl;

    sum_time=0.0;
    for(int i=0;i<LOOP;i++){
        init();
        gettimeofday(&start,NULL);//开始时间
        Gauss_NEON_qi();
        gettimeofday(&end,NULL);//结束时间
        sum_time +=((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000;
    }
    cout<<"Gauss_NEON_qi:"<<(sum_time/LOOP)<<"ms"<<endl;

   
    return 0;
} 


