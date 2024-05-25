#include <iostream>
#include <sys/time.h>
using namespace std;
 
 
const int N = 8388608;
int a[N];
int LOOP = 1000;
 
void init()
{
    for (int i = 0; i < N; i++)
        a[i] = i;
}
 
void ordinary()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int l=0;l<LOOP;l++)
    {
        int sum = 0;
        for (int i = 0; i < N; i++){
            sum += a[i];
        }
            
    }
    gettimeofday(&end,NULL);
    cout<<"ordinary:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
}
 
//多链路式
void optimize1()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int l=0;l<LOOP;l++)
    {
        int sum1 = 0, sum2 = 0;
        for(int i=0;i<N-1; i+=2){
            sum1+=a[i];
            sum2+= a[i+1];

        }       
        int sum = sum1 + sum2;
    }
    gettimeofday(&end,NULL);
    cout<<"ordinary:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
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
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int l=0;l<LOOP;l++)
    {
        recursion(N);
    }
   gettimeofday(&end,NULL);
   cout<<"ordinary:"<<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000/LOOP<<"ms"<<endl;
}
//双重循环
void optimize3()
{
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    for(int l=0;l<LOOP;l++)
    {
        for(int m=N;m>1;m/=2){//logN个步骤
            for(int i=0;i<m/2;i++){
                a[i]=a[i*2]+a[i*2+1];//相邻元素相加连续存储到数组的最前面
            }//a[0]为最终结果
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
    optimize1();
    init();
    optimize2();
    init();
    optimize3();
    return 0;

}