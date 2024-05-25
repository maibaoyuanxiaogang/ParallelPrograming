#include<iostream>
#include<windows.h>
using namespace std;
//数据规模
const int N=8388608;
//N个数的数组
int a[N];
//循环圈数
int LOOP=1000;
//初始化数组，给定N个数的值
void init()
{
    for(int i=0;i<N;i++){
        a[i]=i;
    }
}

//多链路式
void lianlu2()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++)
    {
        int sum1 = 0, sum2 = 0;
        for(int i=0;i<N-1; i+=2){
            sum1+=a[i];
            sum2+= a[i+1];
        }
        int sum = sum1 + sum2;
    }
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"lianlu2:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}

void lianlu4()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++)
    {
        int sum1 = 0, sum2 = 0,sum3=0,sum4=0;
        for(int i=0;i<N-1; i+=4){
            sum1+=a[i];
            sum2+= a[i+1];
            sum3+= a[i+2];
            sum4+= a[i+3];
        }
        int sum = sum1 + sum2+sum3+sum4;

    }
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"lianlu4:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}
void lianlu8()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++)
    {
        int sum1 = 0, sum2 = 0,sum3=0,sum4=0,sum5=0,sum6=0,sum7=0,sum8=0;
        for(int i=0;i<N-1; i+=8){
            sum1+=a[i];
            sum2+= a[i+1];
            sum3+= a[i+2];
            sum4+= a[i+3];
            sum5+= a[i+4];
            sum6+= a[i+5];
            sum7+= a[i+6];
            sum8+= a[i+7];
        }
        int sum = sum1 + sum2+sum3+sum4+sum5+sum6+sum7+sum8;

    }
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"lianlu8:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}
void lianlu16()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++)
    {
        int sum1 = 0, sum2 = 0,sum3=0,sum4=0,sum5=0,sum6=0,sum7=0,sum8=0,sum9=0,sum10=0,sum11=0,sum12=0,sum13=0,sum14=0,sum15=0,sum16=0;
        for(int i=0;i<N-1; i+=16){
            sum1+=a[i];
            sum2+= a[i+1];
            sum3+= a[i+2];
            sum4+= a[i+3];
            sum5+= a[i+4];
            sum6+= a[i+5];
            sum7+= a[i+6];
            sum8+= a[i+7];
            sum9+= a[i+8];
            sum10+= a[i+9];
            sum11+= a[i+10];
            sum12+= a[i+11];
            sum13+= a[i+12];
            sum14+= a[i+13];
            sum15+= a[i+14];
            sum16+= a[i+15];
        }
        int sum = sum1 + sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9 + sum10+sum11+sum12+sum13+sum14+sum15+sum16;
    }
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"lianlu16:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}
int main()
{
    init();
    lianlu2();
    init();
    lianlu4();
    init();
    lianlu8();
    init();
    lianlu16();
    return 0;


}
