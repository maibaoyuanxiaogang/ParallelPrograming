#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<map>
#include<windows.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include<pthread.h>
#include<omp.h>
using namespace std;


const int maxsize = 3000;
const int maxrow = 60000; //3000*32>90000 ,最多存贮列数90000的被消元行矩阵60000行
const int numBasis = 100000;   //最多存储90000*100000的消元子

pthread_mutex_t lock;  //写入消元子时需要加锁


long long head, tail, freq;


map<int, int*>ans;			//答案

fstream RowFile("被消元行.txt", ios::in | ios::out);
fstream BasisFile("消元子.txt", ios::in | ios::out);


int gRows[maxrow][maxsize];   //被消元行最多60000行，3000列
int gBasis[numBasis][maxsize];  //消元子最多40000行，3000列

int ifBasis[numBasis] = { 0 };

void reset() {

	memset(gRows, 0, sizeof(gRows));
	memset(gBasis, 0, sizeof(gBasis));
	memset(ifBasis, 0, sizeof(ifBasis));
	RowFile.close();
	BasisFile.close();
	RowFile.open("被消元行.txt", ios::in | ios::out);
	BasisFile.open("消元子.txt", ios::in | ios::out);


	ans.clear();
}

//读取消元子
int readBasis() {
	for (int i = 0; i < numBasis; i++) {
		if (BasisFile.eof()) {
			//cout << "读取消元子" << i - 1 << "行" << endl;
			return i - 1;
		}
		string tmp;
		bool flag = false;
		int row = 0;
		getline(BasisFile, tmp);
		stringstream s(tmp);
		int pos;
		while (s >> pos) {

			if (!flag) {
				row = pos;
				flag = true;
				//iToBasis.insert(pair<int, int*>(row, gBasis[row]));
				ifBasis[row] = 1;
			}
			int index = pos / 32;
			int offset = pos % 32;
			gBasis[row][index] = gBasis[row][index] | (1 << offset);
		}
		flag = false;
		row = 0;
	}
}

//读取被消元行
int readRowsFrom(int pos) {
	if (RowFile.is_open())
		RowFile.close();
	RowFile.open("被消元行.txt", ios::in | ios::out);
	memset(gRows, 0, sizeof(gRows));   //重置为0
	string line;
	for (int i = 0; i < pos; i++) {       //读取pos前的无关行
		getline(RowFile, line);
	}
	for (int i = pos; i < pos + maxrow; i++) {
		int tmp;
		getline(RowFile, line);
		if (line.empty()) {
			//cout << "读取被消元行 " << i << " 行" << endl;
			return i;   //返回读取的行数
		}
		bool flag = false;
		stringstream s(line);
		while (s >> tmp) {
			int index = tmp / 32;
			int offset = tmp % 32;
			gRows[i - pos][index] = gRows[i - pos][index] | (1 << offset);
			flag = true;
		}
	}
	cout << "read max rows" << endl;
	return -1;  //成功读取maxrow行

}

//寻找第row行被消元行的首项
int findfirst(int row) {
	int first;
	for (int i = maxsize - 1; i >= 0; i--) {
		if (gRows[row][i] == 0)
			continue;
		else {
			int pos = i * 32;
			int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (gRows[row][i] & (1 << k))
				{
					offset = k;
					break;
				}
			}
			first = pos + offset;
			return first;
		}
	}
	return -1;
}



void writeResult(ofstream& out) {
	for (auto it = ans.rbegin(); it != ans.rend(); it++) {
		int* result = it->second;
		int max = it->first / 32 + 1;
		for (int i = max; i >= 0; i--) {
			if (result[i] == 0)
				continue;
			int pos = i * 32;
			//int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (result[i] & (1 << k)) {
					out << k + pos << " ";
				}
			}
		}
		out << endl;
	}
}

//串行
void GE() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //读取被消元行

	int num = (flag == -1) ? maxrow : flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int i = 0; i < num; i++) {
		while (findfirst(i)!= -1) {     //存在首项
			int first =findfirst(i);      //first是首项
			if (ifBasis[first]==1) {  //存在首项为first消元子
				//int* basis = iToBasis.find(first)->second;  //找到该消元子的数组
				for (int j = 0; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];     //进行异或消元

				}
			}
			else {   //升级为消元子
				for (int j = 0; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				break;
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "串行:" << (tail - head) * 1000 / freq << "ms" << endl;

}

//SSE
void SSE_GE() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //读取被消元行
	int num = (flag == -1) ? maxrow : flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int i = 0; i < num; i++) {
		while (findfirst(i) != -1) {
			int first = findfirst(i);
			if (ifBasis[first]==1) {  //存在该消元子
				//int* basis = iToBasis.find(first)->second;
				int j = 0;
				for (; j + 4 < maxsize; j += 4) {
					__m128i vij = _mm_loadu_si128((__m128i*) & gRows[i][j]);
					__m128i vj = _mm_loadu_si128((__m128i*) & gBasis[first][j]);
					__m128i vx = _mm_xor_si128(vij, vj);
					_mm_storeu_si128((__m128i*) & gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
			}
			else {
				int j = 0;
				for (; j + 4 < maxsize; j += 4) {
					__m128i vij = _mm_loadu_si128((__m128i*) & gRows[i][j]);
					_mm_storeu_si128((__m128i*) & gBasis[first][j], vij);
				}
				for (; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				break;

			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "SSE串行:" << (tail - head) * 1000 / freq << "ms" << endl;
}


//AVX
void AVX_GE() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //读取被消元行
	int num = (flag == -1) ? maxrow : flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int i = 0; i < num; i++) {
		while (findfirst(i) != -1) {
			int first = findfirst(i);
			if (ifBasis[first]==1) {  //存在该消元子
				//int* basis = iToBasis.find(first)->second;
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
			}
			else {
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					_mm256_storeu_si256((__m256i*) & gBasis[first][j], vij);
				}
				for (; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				break;

			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "AVX串行:" << (tail - head) * 1000 / freq << "ms" << endl;
}


//AVX512
void AVX512_GE() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //读取被消元行
	int num = (flag == -1) ? maxrow : flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int i = 0; i < num; i++) {
		while (findfirst(i) != -1) {
			int first = findfirst(i);
			if (ifBasis[first]==1) {  //存在该消元子
				//int* basis = iToBasis.find(first)->second;
				int j = 0;
				for (; j + 16 < maxsize; j += 16) {
					__m512i vij = _mm512_loadu_si512((__m512i*) & gRows[i][j]);
					__m512i vj = _mm512_loadu_si512((__m512i*) & gBasis[first][j]);
					__m512i vx = _mm512_xor_si512(vij, vj);
					_mm512_storeu_si512((__m512i*) & gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
			}
			else {
				int j = 0;
				for (; j + 16 < maxsize; j += 16) {
					__m512i vij = _mm512_loadu_si512((__m512i*) & gRows[i][j]);
					_mm512_storeu_si512((__m512i*) & gBasis[first][j], vij);
				}
				for (; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				break;

			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "AVX512:" << (tail - head) * 1000 / freq << "ms" << endl;
}





int main() {

		ofstream out("消元结果.txt");
		ofstream out1("消元结果(AVX).txt");
		ofstream out2("消元结果(AVX512).txt");


		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

		readBasis();
		GE();
		writeResult(out);

		reset();


		readBasis();
		SSE_GE();
		writeResult(out1);

		reset();



		readBasis();
		AVX_GE();
		writeResult(out1);

		reset();


		readBasis();
		AVX512_GE();
		writeResult(out1);

		reset();




		out.close();
		out1.close();
		out2.close();

}
