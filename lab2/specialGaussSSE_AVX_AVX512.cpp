#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<sstream>
#include<math.h>
#include<Windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX��AVX2

using namespace std;

//���ַ���ת��Ϊ01����
void string_to_num(string str, int row, int l, int** arr) {
    string s;
    int a;
    stringstream ss(str);//���ַ���strת��Ϊstringstream���͵Ķ���ss �����ַ�����Ϊ����ĳ�ʼ��ֵ
    //����ת�����ַ��������������������ss>>�����ַ����ж�ȡֵ��ss<<�����ַ�����д��ֵ
    while (ss >> s) {
        stringstream ts;
        ts << s;
        ts >> a;
        arr[row][l - a - 1] = 1; //��01�����е�Ԫ������
    }
}

// ��ȡ01�����е�һ��Ϊ1��Ԫ�ص��±�
int get_first_1(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == 1)
            return size - 1 - i; //��Ϊ���ұ�����0
        else
            continue;
    }
    return -1;
}

// �ж��Ƿ���ڶ�Ӧ����Ԫ��
int _exist(int** E, int* Ed, int row, int line) {
    for (int i = 0; i < row; i++) {
        if (get_first_1(E[i], line) == get_first_1(Ed, line))
            return i;
    }
    return -1;
}

// �����˹��Ԫ���������㷨
void special_Gauss(int** E, int** Ed, int row, int rowd, int line) {
    int count = row - rowd; // ��Ϊ֮ǰrow+=rowd�����ڴ�ԭ����row����һ�п�ʼ
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < rowd; i++) { // ��Ed��ÿһ�н�����Ԫ
        while (get_first_1(Ed[i], line) != -1) { // ������Ԫ�в�ȫΪ0ʱ
            int exist_or_not = _exist(E, Ed[i], row, line); // �ж���û�ж�Ӧ����Ԫ������֮��Ԫ
            if (exist_or_not != -1) { //�ܼ�����Ԫ

                int k;
                for (k = 0; k < line; k++) {
                    Ed[i][k] = Ed[i][k] ^ E[exist_or_not][k]; // ʹ�����������ʵ����Ԫ
                }
            }
            else { // �����ڶ�Ӧ����Ԫ�ӣ�����������Ϊ��Ԫ��
                for (int k = 0; k < line; k++) {
                    E[count][k] = Ed[i][k];
                }
                count++;
                break; // ����ѭ��������һ�н�����Ԫ
            }
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "SpecialGauss:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}


//SSE
void special_Gauss_SSE(int** E, int** Ed, int row, int rowd, int line){
    int count = row - rowd;
	long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int i = 0; i < rowd; i++) {
		while (get_first_1(Ed[i], line) != -1) {
			int exist_or_not = _exist(E, Ed[i], row, line);
			if (exist_or_not != -1) {
				int k;
				for (k = 0; k + 4 <= line; k += 4) {
					//Ed[i][k] = Ed[i][k] ^ E[exist_or_not][k];
					__m128i t1 = _mm_loadu_si128((__m128i*)(Ed[i] + k));
					__m128i t2 = _mm_loadu_si128((__m128i*)(E[exist_or_not] + k));
					t1 = _mm_xor_si128(t1, t2);
					_mm_storeu_si128((__m128i*)(Ed[i] + k), t1);
				}
				for (; k < line; k ++) {
					Ed[i][k] = Ed[i][k] ^ E[exist_or_not][k];
				}

			}
			else {
                int k;
                for(k=0;k+4<=line;k+=4){
                    __m128i t1 = _mm_loadu_si128((__m128i*)(Ed[i] + k));
                    _mm_storeu_si128((__m128i*)(E[count] + k), t1);

                }
				for (; k < line; k++) {
					E[count][k] = Ed[i][k];
				}
				count++;
				break;
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "SSE:" << (tail - head) * 1000.0 / freq  << "ms" << endl;

}

//AVX
void special_Gauss_AVX(int** E, int** Ed, int row, int rowd, int line) {
	int count = row - rowd;
	long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int i = 0; i < rowd; i++) {
		while (get_first_1(Ed[i], line) != -1) {
			int exist_or_not = _exist(E, Ed[i], row, line);
			if (exist_or_not != -1) {
				int k;
				for (k = 0; k + 8 <= line; k += 8) {
					//Ed[i][k] = Ed[i][k] ^ E[exist_or_not][k];
					__m256i t1 = _mm256_loadu_si256((__m256i*)(Ed[i] + k));
					__m256i t2 = _mm256_loadu_si256((__m256i*)(E[exist_or_not] + k));
					t1 = _mm256_xor_si256(t1, t2);
					_mm256_storeu_si256((__m256i*)(Ed[i] + k), t1);
				}
				for (; k < line; k++) {
					Ed[i][k] = Ed[i][k] ^ E[exist_or_not][k];
				}
			}
			else {
                int k;
                for (k = 0; k + 8 <= line; k += 8) {
					__m256i t1 = _mm256_loadu_si256((__m256i*)(Ed[i] + k));
					_mm256_storeu_si256((__m256i*)(E[count] + k), t1);
				}

				for (; k < line; k++) {
					E[count][k] = Ed[i][k];
				}
				count++;
				break;
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "AVX:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}

//AVX512
void special_Gauss_AVX512(int** E, int** Ed, int row, int rowd, int line) {
	int count = row - rowd;
	long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int i = 0; i < rowd; i++) {
		while (get_first_1(Ed[i], line) != -1) {
			int exist_or_not = _exist(E, Ed[i], row, line);
			if (exist_or_not != -1) {
				int k;
				for (k = 0; k + 16 <= line; k += 16) {
					//Ed[i][k] = Ed[i][k] ^ E[exist_or_not][k];
					__m512i t1 = _mm512_loadu_si512((__m512i*)(Ed[i] + k));
					__m512i t2 = _mm512_loadu_si512((__m512i*)(E[exist_or_not] + k));
					t1 = _mm512_xor_si512(t1, t2);
					_mm512_storeu_si512((__m512i*)(Ed[i] + k), t1);
				}
				for (; k < line; k++) {
					Ed[i][k] = Ed[i][k] ^ E[exist_or_not][k];
				}

			}
			else {
                int k;
                for (k = 0; k + 16 <= line; k += 16) {
					__m512i t1 = _mm512_loadu_si512((__m512i*)(Ed[i] + k));
					_mm512_storeu_si512((__m512i*)(E[count] + k), t1);
				}

				for (; k < line; k++) {
					E[count][k] = Ed[i][k];
				}
				count++;
				break;
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "AVX512:" << (tail - head) * 1000.0 / freq << "ms" << endl;
}



int main() {
    ifstream eliminate;//��Ԫ��
    ifstream eliminated;//����Ԫ��
    ofstream result;//���

    int row=1226;//��Ԫ�ӵ�����
    int line=2362;//����
    int rowd=453;//����Ԫ�е�����
    int lined=2362;

    row += rowd;
    //��Ԫ�����ʼ��
    int** E = new int* [row];
    for (int i = 0; i < row; i++)
        E[i] = new int[line];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < line; j++) {
            E[i][j] = 0;
        }
    }
    //����Ԫ�����ʼ��
    int** Ed = new int* [rowd];
    for (int i = 0; i < rowd; i++)
        Ed[i] = new int[lined];
    for (int i = 0; i < rowd; i++) {
        for (int j = 0; j < lined; j++) {
            Ed[i][j] = 0;
        }
    }

    eliminate.open("E:\\Gaussdata\\data\\5_2362_1226_453\\��Ԫ��.txt", ios::in);
    if (!eliminate.is_open()) {
        cout << "��Ԫ���ļ���ʧ��" << endl;
        return 1;
    }
    vector<string> elte;
    string temp1;
    while (getline(eliminate, temp1))
        elte.push_back(temp1);
    eliminate.close();
    for (int i = 0; i < elte.size(); i++)
        string_to_num(elte[i], i, line, E);


    eliminated.open("E:\\Gaussdata\\data\\5_2362_1226_453\\����Ԫ��.txt", ios::in);
    if (!eliminated.is_open()) {
        cout << "����Ԫ���ļ���ʧ��" << endl;
        return 1;
    }
    vector<string> elted;
    string temp2;
    while (getline(eliminated, temp2))
        elted.push_back(temp2);
    eliminated.close();
    for (int i = 0; i < elted.size(); i++)
        string_to_num(elted[i], i, lined, Ed);
    //��ʱ�Ѿ�������01����


    special_Gauss(E, Ed, row, rowd, line);
    elte.clear();
    elted.clear();



    //��Ԫ�����ʼ��
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < line; j++) {
            E[i][j] = 0;
        }
    }
    //����Ԫ�����ʼ��
    for (int i = 0; i < rowd; i++) {
        for (int j = 0; j < lined; j++) {
            Ed[i][j] = 0;
        }
    }
    eliminate.open("E:\\Gaussdata\\data\\5_2362_1226_453\\��Ԫ��.txt", ios::in);
    if (!eliminate.is_open()) {
        cout << "��Ԫ���ļ���ʧ��" << endl;
        return 1;
    }

    string temp11;
    while (getline(eliminate, temp11))
        elte.push_back(temp11);
    eliminate.close();
    for (int i = 0; i < elte.size(); i++)
        string_to_num(elte[i], i, line, E);


    eliminated.open("E:\\Gaussdata\\data\\5_2362_1226_453\\����Ԫ��.txt", ios::in);
    if (!eliminated.is_open()) {
        cout << "����Ԫ���ļ���ʧ��" << endl;
        return 1;
    }

    string temp22;
    while (getline(eliminated, temp22))
        elted.push_back(temp22);
    eliminated.close();
    for (int i = 0; i < elted.size(); i++)
        string_to_num(elted[i], i, lined, Ed);
    //��ʱ�Ѿ�������01����
    special_Gauss_SSE(E,Ed,row,rowd,line);
    elte.clear();
    elted.clear();



    //��Ԫ�����ʼ��
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < line; j++) {
            E[i][j] = 0;
        }
    }
    //����Ԫ�����ʼ��
    for (int i = 0; i < rowd; i++) {
        for (int j = 0; j < lined; j++) {
            Ed[i][j] = 0;
        }
    }
    eliminate.open("E:\\Gaussdata\\data\\5_2362_1226_453\\��Ԫ��.txt", ios::in);
    if (!eliminate.is_open()) {
        cout << "��Ԫ���ļ���ʧ��" << endl;
        return 1;
    }

    string temp111;
    while (getline(eliminate, temp111))
        elte.push_back(temp111);
    eliminate.close();
    for (int i = 0; i < elte.size(); i++)
        string_to_num(elte[i], i, line, E);


    eliminated.open("E:\\Gaussdata\\data\\5_2362_1226_453\\����Ԫ��.txt", ios::in);
    if (!eliminated.is_open()) {
        cout << "����Ԫ���ļ���ʧ��" << endl;
        return 1;
    }

    string temp222;
    while (getline(eliminated, temp222))
        elted.push_back(temp222);
    eliminated.close();
    for (int i = 0; i < elted.size(); i++)
        string_to_num(elted[i], i, lined, Ed);
    //��ʱ�Ѿ�������01����
    special_Gauss_AVX(E, Ed, row, rowd, line);
    elte.clear();
    elted.clear();



    //��Ԫ�����ʼ��
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < line; j++) {
            E[i][j] = 0;
        }
    }
    //����Ԫ�����ʼ��
    for (int i = 0; i < rowd; i++) {
        for (int j = 0; j < lined; j++) {
            Ed[i][j] = 0;
        }
    }
    eliminate.open("E:\\Gaussdata\\data\\5_2362_1226_453\\��Ԫ��.txt", ios::in);
    if (!eliminate.is_open()) {
        cout << "��Ԫ���ļ���ʧ��" << endl;
        return 1;
    }

    string temp1111;
    while (getline(eliminate, temp1111))
        elte.push_back(temp1111);
    eliminate.close();
    for (int i = 0; i < elte.size(); i++)
        string_to_num(elte[i], i, line, E);


    eliminated.open("E:\\Gaussdata\\data\\5_2362_1226_453\\����Ԫ��.txt", ios::in);
    if (!eliminated.is_open()) {
        cout << "����Ԫ���ļ���ʧ��" << endl;
        return 1;
    }

    string temp2222;
    while (getline(eliminated, temp2222))
        elted.push_back(temp2222);
    eliminated.close();
    for (int i = 0; i < elted.size(); i++)
        string_to_num(elted[i], i, lined, Ed);
    //��ʱ�Ѿ�������01����
    special_Gauss_AVX512(E, Ed, row, rowd, line);

    result.open("E:\\Gaussdata\\data\\5_2362_1226_453\\result.txt", ios::out);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < line; j++)
        {
            result << E[i][j];
        }
        result << endl;
    }
    // �ͷ��ڴ�
    for (int i = 0; i < row; i++)
        delete[] E[i];
    delete[] E;
    for (int i = 0; i < rowd; i++)
        delete[] Ed[i];
    delete[]Ed;
    return 0;
}
