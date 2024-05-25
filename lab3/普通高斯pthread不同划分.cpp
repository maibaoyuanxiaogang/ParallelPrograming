#include <iostream>
#include <windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX
#include <pthread.h>
#include <semaphore.h>

#define n 2500
#define thread_count 7
using namespace std;

static float A[n][n];
int id[thread_count];
sem_t sem_parent;
pthread_barrier_t childbarrier_row;
pthread_barrier_t childbarrier_col;
long long head, tail, freq;

void init()
{
    //生成一个上三角矩阵，对角线为1.0
    for(int i=0;i<n;i++){
        for(int j=0;j<i;j++){
            A[i][j]=0.0;
        }
        A[i][i]=1.0;
        for(int k=i+1;k<n;k++){
            //A[i][k]=(rand()%1000)/100.0;
            A[i][k]=rand()%10;
        }
    }
    //构造一个更为复杂的矩阵，但是因为只是加上行，通过初等行变化可还原成上三角矩阵
    //每一行的元素都加上其前面几行的元素
    for(int k=0;k<n;k++){
        for(int i=k+1;i<n;i++){
            for(int j=0;j<n;j++){
                A[i][j]+=A[k][j];
            }
        }
    }

}

void printA()
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
            cout<<A[i][j]<<" ";
        cout<<endl;
    }
}

void normal_gauss()
{
    for(int k=0;k<n;k++)
    {
        for(int j=k+1;j<n;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        A[k][k]=1;
        for(int i=k+1;i<n;i++)
        {
            for(int j=k+1;j<n;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0;
        }
    }
}


void * processbycol(void * ID)
{
    int * threadid= (int*)ID;
    for(int k=0;k<n;k++)
    {
        int begin=k+1+*threadid*((n-k-1)/thread_count);
        int end=begin+(n-k-1)/thread_count;
        if(end>n)
            end=n;
        for(int i=begin;i<end;i++)
        {
            A[k][i]=A[k][i]/A[k][k];
        }
        for(int i=k+1;i<n;i++)
        {
            for(int j=begin;j<end;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
        }
        sem_post(&sem_parent);
        pthread_barrier_wait(&childbarrier_col);
    }
    pthread_exit(NULL);
}

void Gauss_pthread_col()
{
    pthread_t threadID[thread_count];
    for(int k=0;k<n;k++)
    {
        if(k==0)
        {
            for(int i=0;i<thread_count;i++)
            {
                pthread_create(&threadID[i],NULL,processbycol,(void*)&id[i]);
            }
        }
        for(int i=0;i<thread_count;i++)
        {
            sem_wait(&sem_parent);
        }
        A[k][k]=1;
        for(int i=k+1;i<n;i++)
            A[i][k]=0;
        pthread_barrier_wait(&childbarrier_col);
    }
    for(int i=0;i<thread_count;i++)
    {
        pthread_join(threadID[i],NULL);
    }
    return;
}

void * processbyrow(void * ID)
{
    int* threadid= (int*)ID;
    for(int k=0;k<n;k++)
    {
        int begin=k+1+*threadid*((n-k-1)/thread_count);
        int end=begin+(n-k-1)/thread_count;
        if(end>n)
            end=n;
        for(int i=begin;i<end;i++)
        {
            for(int j=k+1;j<n;j++)
            {
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0;
        }
        sem_post(&sem_parent);
        pthread_barrier_wait(&childbarrier_row);
    }
    pthread_exit(NULL);
}

void Gauss_pthread_row()
{
    pthread_t threadID[thread_count];
    for(int k=0;k<n;k++)
    {
        for(int j=k+1;j<n;j++)
        {
            A[k][j]=A[k][j]/A[k][k];
        }
        A[k][k]=1;
        if(k==0)
        {
            for(int i=0;i<thread_count;i++)
            {
                pthread_create(&threadID[i],NULL,processbyrow,(void*)&id[i]);
            }
        }
        else
            pthread_barrier_wait(&childbarrier_row);
        for(int i=0;i<thread_count;i++)
        {
            sem_wait(&sem_parent);
        }
    }
    pthread_barrier_wait(&childbarrier_row);
    for(int i=0;i<thread_count;i++)
    {
        pthread_join(threadID[i],NULL);
    }
    return;
}




int main()
{
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    pthread_barrier_init(&childbarrier_row, NULL,thread_count+1);
    pthread_barrier_init(&childbarrier_col,NULL, thread_count+1);
    sem_init(&sem_parent, 0, 0);
    for(int i=0;i<thread_count;i++)
        id[i]=i;

    init();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    normal_gauss();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout<<"普通高斯"<<(tail-head)*1000.0/freq<<"ms"<< endl;


    init();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    Gauss_pthread_row();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout<<"行划分"<<(tail-head)*1000.0/freq<<"ms"<< endl;

    init();
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    Gauss_pthread_col();
    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    cout<<"列划分"<<(tail-head)*1000.0/freq<<"ms"<< endl;


    sem_destroy(&sem_parent);
    pthread_barrier_destroy(&childbarrier_col);
    pthread_barrier_destroy(&childbarrier_row);
	return 0;
}
