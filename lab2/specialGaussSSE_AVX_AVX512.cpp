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
#include <immintrin.h> //AVX、AVX2

using namespace std;

//将字符串转换为01矩阵
void string_to_num(string str, int row, int l, int** arr) {
    string s;
    int a;
    stringstream ss(str);//将字符串str转换为stringstream类型的对象ss 并将字符串作为对象的初始化值
    //这种转换对字符串进行输入输出操作，ss>>来从字符串中读取值，ss<<来向字符串中写入值
    while (ss >> s) {
        stringstream ts;
        ts << s;
        ts >> a;
        arr[row][l - a - 1] = 1; //将01矩阵中的元素填入
    }
}

// 获取01矩阵中第一个为1的元素的下标
int get_first_1(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == 1)
            return size - 1 - i; //因为最右边是列0
        else
            continue;
    }
    return -1;
}

// 判断是否存在对应的消元子
int _exist(int** E, int* Ed, int row, int line) {
    for (int i = 0; i < row; i++) {
        if (get_first_1(E[i], line) == get_first_1(Ed, line))
            return i;
    }
    return -1;
}

// 特殊高斯消元方法串行算法
void special_Gauss(int** E, int** Ed, int row, int rowd, int line) {
    int count = row - rowd; // 因为之前row+=rowd，现在从原来的row的下一行开始
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < rowd; i++) { // 对Ed中每一行进行消元
        while (get_first_1(Ed[i], line) != -1) { // 当被消元行不全为0时
            int exist_or_not = _exist(E, Ed[i], row, line); // 判断有没有对应的消元子能与之消元
            if (exist_or_not != -1) { //能继续消元

                int k;
                for (k = 0; k < line; k++) {
                    Ed[i][k] = Ed[i][k] ^ E[exist_or_not][k]; // 使用异或运算来实现消元
                }
            }
            else { // 不存在对应的消元子，将该行升级为消元子
                for (int k = 0; k < line; k++) {
                    E[count][k] = Ed[i][k];
                }
                count++;
                break; // 跳出循环，对下一行进行消元
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
    ifstream eliminate;//消元子
    ifstream eliminated;//被消元行
    ofstream result;//结果

    int row=1226;//消元子的行数
    int line=2362;//列数
    int rowd=453;//被消元行的行数
    int lined=2362;

    row += rowd;
    //消元矩阵初始化
    int** E = new int* [row];
    for (int i = 0; i < row; i++)
        E[i] = new int[line];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < line; j++) {
            E[i][j] = 0;
        }
    }
    //被消元矩阵初始化
    int** Ed = new int* [rowd];
    for (int i = 0; i < rowd; i++)
        Ed[i] = new int[lined];
    for (int i = 0; i < rowd; i++) {
        for (int j = 0; j < lined; j++) {
            Ed[i][j] = 0;
        }
    }

    eliminate.open("E:\\Gaussdata\\data\\5_2362_1226_453\\消元子.txt", ios::in);
    if (!eliminate.is_open()) {
        cout << "消元子文件打开失败" << endl;
        return 1;
    }
    vector<string> elte;
    string temp1;
    while (getline(eliminate, temp1))
        elte.push_back(temp1);
    eliminate.close();
    for (int i = 0; i < elte.size(); i++)
        string_to_num(elte[i], i, line, E);


    eliminated.open("E:\\Gaussdata\\data\\5_2362_1226_453\\被消元行.txt", ios::in);
    if (!eliminated.is_open()) {
        cout << "被消元行文件打开失败" << endl;
        return 1;
    }
    vector<string> elted;
    string temp2;
    while (getline(eliminated, temp2))
        elted.push_back(temp2);
    eliminated.close();
    for (int i = 0; i < elted.size(); i++)
        string_to_num(elted[i], i, lined, Ed);
    //此时已经建立好01矩阵


    special_Gauss(E, Ed, row, rowd, line);
    elte.clear();
    elted.clear();



    //消元矩阵初始化
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < line; j++) {
            E[i][j] = 0;
        }
    }
    //被消元矩阵初始化
    for (int i = 0; i < rowd; i++) {
        for (int j = 0; j < lined; j++) {
            Ed[i][j] = 0;
        }
    }
    eliminate.open("E:\\Gaussdata\\data\\5_2362_1226_453\\消元子.txt", ios::in);
    if (!eliminate.is_open()) {
        cout << "消元子文件打开失败" << endl;
        return 1;
    }

    string temp11;
    while (getline(eliminate, temp11))
        elte.push_back(temp11);
    eliminate.close();
    for (int i = 0; i < elte.size(); i++)
        string_to_num(elte[i], i, line, E);


    eliminated.open("E:\\Gaussdata\\data\\5_2362_1226_453\\被消元行.txt", ios::in);
    if (!eliminated.is_open()) {
        cout << "被消元行文件打开失败" << endl;
        return 1;
    }

    string temp22;
    while (getline(eliminated, temp22))
        elted.push_back(temp22);
    eliminated.close();
    for (int i = 0; i < elted.size(); i++)
        string_to_num(elted[i], i, lined, Ed);
    //此时已经建立好01矩阵
    special_Gauss_SSE(E,Ed,row,rowd,line);
    elte.clear();
    elted.clear();



    //消元矩阵初始化
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < line; j++) {
            E[i][j] = 0;
        }
    }
    //被消元矩阵初始化
    for (int i = 0; i < rowd; i++) {
        for (int j = 0; j < lined; j++) {
            Ed[i][j] = 0;
        }
    }
    eliminate.open("E:\\Gaussdata\\data\\5_2362_1226_453\\消元子.txt", ios::in);
    if (!eliminate.is_open()) {
        cout << "消元子文件打开失败" << endl;
        return 1;
    }

    string temp111;
    while (getline(eliminate, temp111))
        elte.push_back(temp111);
    eliminate.close();
    for (int i = 0; i < elte.size(); i++)
        string_to_num(elte[i], i, line, E);


    eliminated.open("E:\\Gaussdata\\data\\5_2362_1226_453\\被消元行.txt", ios::in);
    if (!eliminated.is_open()) {
        cout << "被消元行文件打开失败" << endl;
        return 1;
    }

    string temp222;
    while (getline(eliminated, temp222))
        elted.push_back(temp222);
    eliminated.close();
    for (int i = 0; i < elted.size(); i++)
        string_to_num(elted[i], i, lined, Ed);
    //此时已经建立好01矩阵
    special_Gauss_AVX(E, Ed, row, rowd, line);
    elte.clear();
    elted.clear();



    //消元矩阵初始化
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < line; j++) {
            E[i][j] = 0;
        }
    }
    //被消元矩阵初始化
    for (int i = 0; i < rowd; i++) {
        for (int j = 0; j < lined; j++) {
            Ed[i][j] = 0;
        }
    }
    eliminate.open("E:\\Gaussdata\\data\\5_2362_1226_453\\消元子.txt", ios::in);
    if (!eliminate.is_open()) {
        cout << "消元子文件打开失败" << endl;
        return 1;
    }

    string temp1111;
    while (getline(eliminate, temp1111))
        elte.push_back(temp1111);
    eliminate.close();
    for (int i = 0; i < elte.size(); i++)
        string_to_num(elte[i], i, line, E);


    eliminated.open("E:\\Gaussdata\\data\\5_2362_1226_453\\被消元行.txt", ios::in);
    if (!eliminated.is_open()) {
        cout << "被消元行文件打开失败" << endl;
        return 1;
    }

    string temp2222;
    while (getline(eliminated, temp2222))
        elted.push_back(temp2222);
    eliminated.close();
    for (int i = 0; i < elted.size(); i++)
        string_to_num(elted[i], i, lined, Ed);
    //此时已经建立好01矩阵
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
    // 释放内存
    for (int i = 0; i < row; i++)
        delete[] E[i];
    delete[] E;
    for (int i = 0; i < rowd; i++)
        delete[] Ed[i];
    delete[]Ed;
    return 0;
}
