#include<iostream>
#include<windows.h>
using namespace std;
//���ݹ�ģ
const int N=8192;
//N����������
int a[N];
//ѭ��Ȧ��
int LOOP=1000;
//��ʼ�����飬����N������ֵ
void init()
{
    for(int i=0;i<N;i++){
        a[i]=i;
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
        int sum=0;
        for(int i=0;i<N;i++){
            sum+=a[i];
        }
    }
    //����ʱ��
    QueryPerformanceCounter((LARGE_INTEGER*) &end);
    cout<<"ordinary:"<<(end-begin)*1000.0/freq/LOOP<<"ms"<<endl;
}

//����·ʽ
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
//�ݹ�,�ⲻ�����Ǹ���˹
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
//˫��ѭ��
void optimize3()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    QueryPerformanceCounter((LARGE_INTEGER*) &begin);
    for(int l=0;l<LOOP;l++)
    {
        for(int m=N;m>1;m/=2){//logN������
            for(int i=0;i<m/2;i++){
                a[i]=a[i*2]+a[i*2+1];//����Ԫ����������洢���������ǰ��
            }//a[0]Ϊ���ս��
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



