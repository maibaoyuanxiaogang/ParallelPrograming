#include <iostream>
#include <Windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX��AVX2

using namespace std;

//��ʾ�����ģ
const int N=4000;
//����
float A[N][N];
//ѭ������
const int LOOP=1;

void init(){
    //����һ�������Ǿ��󣬶Խ���Ϊ1.0
    for(int i=0;i<N;i++){
        for(int j=0;j<i;j++){
            A[i][j]=0.0;
        }
        A[i][i]=1.0;
        for(int k=i+1;k<N;k++){
            //A[i][k]=(rand()%1000)/100.0;
            A[i][k]=rand()%10;
        }
    }


    //����һ����Ϊ���ӵľ��󣬵�����Ϊֻ�Ǽ����У�ͨ�������б仯�ɻ�ԭ�������Ǿ���
    //ÿһ�е�Ԫ�ض�������ǰ�漸�е�Ԫ��
    for(int k=0;k<N;k++){
        for(int i=k+1;i<N;i++){
            for(int j=0;j<N;j++){
                A[i][j]+=A[k][j];
            }
        }
    }
}

//��ͨ��˹��ȥ�������㷨
void Gauss(){
    for(int k=0;k<N;k++){
        for(int j=k+1;j<N;j++){
            A[k][j]/=A[k][k];
        }
        A[k][k]=1.0;//���Խ���Ԫ�ر�Ϊ1

        for(int i=k+1;i<N;i++){
            for(int j=k+1;j<N;j++){
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0.0;
        }

    }

}


//SSE
void Gauss_SSE(){
    for(int k=0;k<N;k++){
        //��A[k][k]���Ƶ�t1���ĸ�λ��
        __m128 v1 = _mm_set1_ps(A[k][k]);
        int j;
        for(j=k+1;j+4<=N;j+=4){
            //A[k][j]/=A[k][k]
            __m128 va = _mm_loadu_ps(A[k] + j);
            va=_mm_div_ps(va,v1);
            _mm_storeu_ps(A[k]+j,va);
        }
        //�������������Ǽ�����������
        for(;j<N;j++){
            A[k][j]/=A[k][k];
        }
        A[k][k]=1.0;

        for(int i=k+1;i<N;i++){
            //�����׷���Ԫ
            __m128 v2 = _mm_set1_ps(A[i][k]);
            int j;
            for(j=k+1;j+4<=N;j+=4){
                //A[i][j]=A[i][j]-A[i][k]*A[k][j];
                __m128 vaij=_mm_loadu_ps(A[i]+j);
                __m128 vakj=_mm_loadu_ps(A[k]+j);
                __m128 vx=_mm_mul_ps(v2,vakj);
                vaij=_mm_sub_ps(vaij,vx);
                _mm_storeu_ps(A[i]+j,vaij);
            }
            //�������������Ǽ�����������
            for(;j<N;j++){
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0.0;

        }
    }

}
//SSE1
void Gauss_SSE1(){
    for(int k=0;k<N;k++){
        //��A[k][k]���Ƶ�t1���ĸ�λ��
        __m128 v1 = _mm_set1_ps(A[k][k]);
        int j;
        for(j=k+1;j+4<=N;j+=4){
            //A[k][j]/=A[k][k]
            __m128 va = _mm_loadu_ps(A[k] + j);
            va=_mm_div_ps(va,v1);
            _mm_storeu_ps(A[k]+j,va);
        }
        //�������������Ǽ�����������
        for(;j<N;j++){
            A[k][j]/=A[k][k];
        }
        A[k][k]=1.0;

        for(int i=k+1;i<N;i++){
            for(int j=k+1;j<N;j++){
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0.0;
        }
    }
}

//SSE2
void Gauss_SSE2(){
    for(int k=0;k<N;k++){
        for(int j=k+1;j<N;j++){
            A[k][j]/=A[k][k];
        }
        A[k][k]=1.0;

        for(int i=k+1;i<N;i++){
            //�����׷���Ԫ
            __m128 v2 = _mm_set1_ps(A[i][k]);
            int j;
            for(j=k+1;j+4<=N;j+=4){
                //A[i][j]=A[i][j]-A[i][k]*A[k][j];
                __m128 vaij=_mm_loadu_ps(A[i]+j);
                __m128 vakj=_mm_loadu_ps(A[k]+j);
                __m128 vx=_mm_mul_ps(v2,vakj);
                vaij=_mm_sub_ps(vaij,vx);
                _mm_storeu_ps(A[i]+j,vaij);
            }
            //�������������Ǽ�����������
            for(;j<N;j++){
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0.0;

        }
    }

}




void Gauss_SSE_qi(){
    for(int k=0;k<N;k++){
        //���ȼ���A[k][k]
        __m128 v1 = _mm_set1_ps(A[k][k]);
        //���м���ֱ������
        int j;
        for(j=k+1;j<N;j++){
            if(((size_t)(A[k]+j))%16==0){
                break;
            }
            A[k][j]/=A[k][k];
        }
        //�����ˣ���ʼ����
        for(;j+4<=N;j+=4){
            //A[k][j]/=A[k][k]
            __m128 va=_mm_load_ps(A[k]+j);
            va=_mm_div_ps(va,v1);
            _mm_store_ps(A[k]+j,va);
        }
        //����4��
        for(;j<N;j++){
            A[k][j]/=A[k][k];
        }
        A[k][k]=1.0;

        for(int i=k+1;i<N;i++){
            __m128 vaik = _mm_set1_ps(A[i][k]);
            //���м���ֱ������
            int j;
            for(j = k + 1; j < N; j++) {
                if(((size_t)(A[i]+j))%16==0){
                    break;
                }
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            for (; j + 4 <= N; j += 4) {
				//A[i][j] = A[i][j] - A[i][k] * A[k][j];
				__m128 vaij = _mm_load_ps(A[i] + j);
				__m128 vakj = _mm_loadu_ps(A[k] + j);

                __m128 vx=_mm_mul_ps(vaik,vakj);
                vaij=_mm_sub_ps(vaij,vx);
                _mm_store_ps(A[i]+j,vaij);

			}

			for(;j<N;j++){
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
			}
			A[i][k]=0.0;

        }
    }
}







//AVX
void Gauss_AVX(){
    for(int k=0;k<N;k++){
        __m256 v1=_mm256_set1_ps(A[k][k]);
        int j;
        for(j=k+1;j+8<=N;j+=8){
            //A[k][j]/=A[k][k]
            __m256 va=_mm256_loadu_ps(A[k]+j);
            va=_mm256_div_ps(va,v1);
            _mm256_storeu_ps(A[k] + j, va);
        }
        for(;j<N;j++){
            A[k][j]/=A[k][k];
        }
        A[k][k]=1.0;

        for(int i=k+1;i<N;i++){
            __m256 vaik = _mm256_set1_ps(A[i][k]);
            int j;
            for(j=k+1;j+8<=N;j+=8){
                //A[i][j]=A[i][j]-A[i][k]*A[k][j]
                __m256 vaij=_mm256_loadu_ps(A[i]+j);
                __m256 vakj=_mm256_loadu_ps(A[k]+j);
                __m256 vx=_mm256_mul_ps(vaik,vakj);
                vaij=_mm256_sub_ps(vaij,vx);
                _mm256_storeu_ps(A[i]+j,vaij);
            }
            for(;j<N;j++){
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0.0;
        }

    }

}


//AVX����
void Gauss_AVX_qi(){
    for(int k=0;k<N;k++){
        __m256 v1=_mm256_set1_ps(A[k][k]);
        int j;
        //����ֱ������
        for(j=0;j<N;j++){
            if(((size_t)(A[k]+j))%32==0){
                break;
            }
            A[k][j]/=A[k][k];
        }
        for(;j+8<=N;j+=8){
            __m256 vakj=_mm256_load_ps(A[k]+j);
            vakj=_mm256_div_ps(vakj,v1);
            _mm256_store_ps(A[k]+j,vakj);
        }
        for(;j<N;j++){
            A[k][j]/=A[k][k];
        }
        A[k][k]=1.0;


        for(int i=k+1;i<N;i++){
            __m256 vaik = _mm256_set1_ps(A[i][k]);
            //���м���ֱ������
            int j;
            for(j = k + 1; j < N; j++) {
                if(((size_t)(A[i]+j))%32==0){
                    break;
                }
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            for (; j + 8 <= N; j += 8) {
				//A[i][j] = A[i][j] - A[i][k] * A[k][j];
				__m256 vaij = _mm256_load_ps(A[i] + j);
				__m256 vakj = _mm256_loadu_ps(A[k] + j);

                __m256 vx=_mm256_mul_ps(vaik,vakj);
                vaij=_mm256_sub_ps(vaij,vx);
                _mm256_store_ps(A[i]+j,vaij);

			}

			for(;j<N;j++){
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
			}
			A[i][k]=0.0;

        }

    }


}

//AVX-512
void Gauss_AVX512(){
    for(int k=0;k<N;k++){
        __m512 v1=_mm512_set1_ps(A[k][k]);
        int j;
        for(j=k+1;j+16<=N;j+=16){
            //A[k][j]/=A[k][k]
            __m512 va=_mm512_loadu_ps(A[k]+j);
            va=_mm512_div_ps(va,v1);
            _mm512_storeu_ps(A[k]+j,va);
        }
        for(;j<N;j++){
            A[k][j]/=A[k][k];
        }
        A[k][k]=1.0;

        for(int i=k+1;i<N;i++){
            __m512 vaik=_mm512_set1_ps(A[i][k]);
            int j;
            for(j=k+1;j+16<=N;j+=16){
                //A[i][j]=A[i][j]-A[i][k]*A[k][j]
                __m512 vaij=_mm512_loadu_ps(A[i]+j);
                __m512 vakj=_mm512_loadu_ps(A[k]+j);
                __m512 vx=_mm512_mul_ps(vaik,vakj);
                vaij=_mm512_sub_ps(vaij,vx);
                _mm512_storeu_ps(A[i]+j,vaij);
            }
            for(;j<N;j++){
                A[i][j]=A[i][j]-A[i][k]*A[k][j];
            }
            A[i][k]=0.0;
        }
    }

}

//AVX512����
void Gauss_AVX512_qi(){
    for(int k=0;k<N;k++){
        __m512 v1=_mm512_set1_ps(A[k][k]);
        int j;
        //����ֱ������
        for(j=0;j<N;j++){
            if(((size_t)(A[k]+j))%64==0){
                break;
            }
            A[k][j]/=A[k][k];
        }
        for(;j+16<=N;j+=16){
            __m512 vakj=_mm512_load_ps(A[k]+j);
            vakj=_mm512_div_ps(vakj,v1);
            _mm512_store_ps(A[k]+j,vakj);
        }
        for(;j<N;j++){
            A[k][j]/=A[k][k];
        }
        A[k][k]=1.0;


        for(int i=k+1;i<N;i++){
            __m512 vaik = _mm512_set1_ps(A[i][k]);
            //���м���ֱ������
            int j;
            for(j = k + 1; j < N; j++) {
                if(((size_t)(A[i]+j))%64==0){
                    break;
                }
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
            }
            for (; j + 16 <= N; j += 16) {
				//A[i][j] = A[i][j] - A[i][k] * A[k][j];
				__m512 vaij = _mm512_load_ps(A[i] + j);
				__m512 vakj = _mm512_loadu_ps(A[k] + j);

                __m512 vx=_mm512_mul_ps(vaik,vakj);
                vaij=_mm512_sub_ps(vaij,vx);
                _mm512_store_ps(A[i]+j,vaij);

			}

			for(;j<N;j++){
                A[i][j] = A[i][j] - A[i][k]*A[k][j];
			}
			A[i][k]=0.0;

        }

    }

}




void print(){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            cout<<A[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;


}





int main()
{
    long long int begin,end,freq;
    double sum_time=0.0;
    for(int i=0;i<LOOP;i++){
        init();
        QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
        //��ʼʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &begin);
        Gauss();
        //����ʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &end);
        sum_time+=(end-begin)*1000.0/freq;

    }
    cout << "Gauss:" << (sum_time/LOOP) << "ms" << endl;

    sum_time=0.0;
    for(int i=0;i<LOOP;i++){
        init();
        QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
        //��ʼʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &begin);
        Gauss_SSE();
        //����ʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &end);
        sum_time+=(end-begin)*1000.0/freq;

    }
    cout << "Gauss_SSE:" << (sum_time/LOOP) << "ms" << endl;

    sum_time=0.0;
    for(int i=0;i<LOOP;i++){
        init();
        QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
        //��ʼʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &begin);
        Gauss_SSE1();
        //����ʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &end);
        sum_time+=(end-begin)*1000.0/freq;

    }
    cout << "Gauss_SSE1:" << (sum_time/LOOP) << "ms" << endl;

    sum_time=0.0;
    for(int i=0;i<LOOP;i++){
        init();
        QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
        //��ʼʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &begin);
        Gauss_SSE2();
        //����ʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &end);
        sum_time+=(end-begin)*1000.0/freq;

    }
    cout << "Gauss_SSE2:" << (sum_time/LOOP) << "ms" << endl;

    sum_time=0.0;
    for(int i=0;i<LOOP;i++){
        init();
        QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
        //��ʼʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &begin);
        Gauss_SSE_qi();
        //����ʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &end);
        sum_time+=(end-begin)*1000.0/freq;

    }
    cout << "Gauss_SSE_qi:" << (sum_time/LOOP) << "ms" << endl;

    sum_time=0.0;
    for(int i=0;i<LOOP;i++){
        init();
        QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
        //��ʼʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &begin);
        Gauss_AVX();
        //����ʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &end);
        sum_time+=(end-begin)*1000.0/freq;

    }
    cout << "Gauss_AVX:" << (sum_time/LOOP) << "ms" << endl;

    sum_time=0.0;
    for(int i=0;i<LOOP;i++){
        init();
        QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
        //��ʼʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &begin);
        Gauss_AVX_qi();
        //����ʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &end);
        sum_time+=(end-begin)*1000.0/freq;

    }
    cout << "Gauss_AVX_qi:" << (sum_time/LOOP) << "ms" << endl;

    sum_time=0.0;
    for(int i=0;i<LOOP;i++){
        init();
        QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
        //��ʼʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &begin);
        Gauss_AVX512();
        //����ʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &end);
        sum_time+=(end-begin)*1000.0/freq;

    }
    cout << "Gauss_AVX512:" << (sum_time/LOOP) << "ms" << endl;

    sum_time=0.0;
    for(int i=0;i<LOOP;i++){
        init();
        QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
        //��ʼʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &begin);
        Gauss_AVX512_qi();
        //����ʱ��
        QueryPerformanceCounter((LARGE_INTEGER*) &end);
        sum_time+=(end-begin)*1000.0/freq;

    }
    cout << "Gauss_AVX512_qi:" << (sum_time/LOOP) << "ms" << endl;







    return 0;
}




