#include<iostream>
#include<windows.h>
using namespace std;
//数据规模
const int N=8192;
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

//平凡算法
void ordinary()
{
    long long int begin,end,freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    //开始时间
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++){
        int sum=0;
        for(int i=0;i<N;i++){
            sum+=a[i];
        }
    }
    //结束时间
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"ordinary:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}

//多链路式
void optimize1()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++)
    {
        int sum1 = 0, sum2 = 0;
        for(int i=0;i<N-1; i+=2)
            sum1+=a[i],sum2+= a[i+1];
        int sum = sum1 + sum2;
    }
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"optimize1:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}
//递归,这不就像那个高斯
void recursion(int N)
{
    if(N==1){
        return;
    }
    else{
        for(int i=0;i<N/2;i++){
            a[i]+=a[N-i-1];
        }
        N=N/2;
        recursion(N);
    }
}
void optimize2()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++)
    {
        recursion(N);

    }
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"optimize2:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}
//双重循环
void optimize3()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++)
    {
        for(int m=N;m>1;m/=2){//logN个步骤
            for(int i=0;i<m/2;i++){
                a[i]=a[i*2]+a[i*2+1];//相邻元素相加连续存储到数组的最前面
            }//a[0]为最终结果
        }

    }
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"optimize3:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}
int main()
{
    init();
    ordinary();
    init();
    optimize1();
    init();
    optimize2();
    init();
    optimize3();
    return 0;


}



