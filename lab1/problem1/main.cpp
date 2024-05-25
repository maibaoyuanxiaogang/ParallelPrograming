#include <iostream>
#include <windows.h>
using namespace std;
//��ʾ�����ģ
const int N=9000;
//����
int a[N];
//����
int b[N][N];
//���
int sum[N];
//ѭ��ִ�д���
const int LOOP=10;

//�����������ʼ��������ȷ����ֵ
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
//ƽ���㷨
void ordinary()
{
    long long int begin,end,freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    //��ʼʱ��
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++){
        for(int i=0;i<N;i++){
            sum[i]=0;
            for(int j=0;j<N;j++){
                sum[i]+=a[j]*b[j][i];
            }
        }
    }
    //����ʱ��
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"ordinary:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;


}
//cache�Ż��㷨
void optimize()
{
    long long int begin,end,freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    //��ʼʱ��
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++){
        for(int i=0;i<N;i++){
            sum[i]=0;
        }
        for(int j=0;j<N;j++){//ÿһ��
            for(int i=0;i<N;i++){//ÿһ��
                sum[i]+=a[j]*b[j][i];
            }
        }

    }
    //����ʱ��
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"optimize:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}
//ѭ��չ���㷨
void unroll()
{
    long long int begin,end,freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    //��ʼʱ��
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++){
        for(int i=0;i<N;i++){
            sum[i]=0;
        }
        for(int j=0;j<N;j+=10){//һ����ȡʮ��
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
            for(int i=0;i<N;i++){//ÿһ��
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
    //����ʱ��
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"unroll:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}

int main()
{
   init();
   ordinary();
   init();
   optimize();
   init();
   unroll();

    return 0;
}

