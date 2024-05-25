#include<iostream>
#include <stdio.h>
#include<sys/time.h>
using namespace std;
//表示矩阵规模
const int N=9000;
//向量
int a[N];
//矩阵
int b[N][N];
//结果
int sum[N];
//循环执行次数
const int LOOP=10;

//向量及矩阵初始化，给定确定的值
void init()
{
    for(int i=0;i<N;i++){
        a[i]=i;
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            b[i][j]=i+j;
        }
    }
}
//平凡算法
void ordinary()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);//开始时间
    for(int l=0;l<LOOP;l++){
        for(int i=0;i<N;i++){
            sum[i]=0;
            for(int j=0;j<N;j++){
                sum[i]+=a[j]*b[j][i];
            }
        }
    }
    gettimeofday(&end,NULL);//结束时间
    cout<<"ordinary:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
}

void optimize()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int l=0;l<LOOP;l++)
    {
        for(int i=0;i<N;i++)
            sum[i]=0;
        for(int j=0;j<N;j++){//每一行
            for(int i=0;i<N;i++){//每一列
                sum[i]+=a[j]*b[j][i];
            }
        }
    }
    gettimeofday(&end,NULL);
    cout<<"ordinary:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
}

void unroll()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int l=0;l<LOOP;l++)
    {
        for(int i=0;i<N;i++)
            sum[i]=0;
        for(int j=0;j<N;j+=10){//一下子取十行
            int row0=j+0;
            int row1=j+1;
            int row2=j+2;
            int row3=j+3;
            int row4=j+4;
            int row5=j+5;
            int row6=j+6;
            int row7=j+7;
            int row8=j+8;
            int row9=j+9;
            for(int i=0;i<N;i++){//每一列
                sum[row0]+=a[row0]*b[row0][i];
                sum[row1]+=a[row1]*b[row1][i];
                sum[row2]+=a[row2]*b[row2][i];
                sum[row3]+=a[row3]*b[row3][i];
                sum[row4]+=a[row4]*b[row4][i];
                sum[row5]+=a[row5]*b[row5][i];
                sum[row6]+=a[row6]*b[row6][i];
                sum[row7]+=a[row7]*b[row7][i];
                sum[row8]+=a[row8]*b[row8][i];
                sum[row9]+=a[row9]*b[row9][i];
            }
        }
    }
    gettimeofday(&end,NULL);
    cout<<"ordinary:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
}

int main()
{
    init();
    ordinary();
    init();
    optimize();
    init();
    unroll();
}
