#include<iostream>
#include <cstdio>
#include <fstream>
#include <sys/stat.h>
#include<string>
#include<vector>
#include<sstream>
#include<math.h>
#include<arm_neon.h>
#include<sys/time.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

void string_to_num(string str,int row,int l, int** arr) {
    string s;
    int a;
    stringstream ss(str);
    while (ss >> s) {
        stringstream ts;
        ts << s;
        ts >> a;
        arr[row][l - a -1] = 1;
        
    }
}

int find1(int* arr,int size) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == 1)
            return size - 1 - i;
        else
            continue;
    }
    return -1;
}

int _exist(int** E, int* Ed, int row, int line) {
    for (int i = 0; i < row; i++) {
        if (find1(E[i], line) == find1(Ed, line))
            return i;
    }
    return -1;
}

void special_Gauss(int** E, int** Ed,int row,int rowd,int line) {
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    int count = row - rowd;
    for (int i = 0; i < rowd; i++) {
        while (find1(Ed[i], line) != -1) {
            int exist_or_not = _exist(E, Ed[i], row, line);
            if (exist_or_not != -1) {
                int k;
                for (k = 0; k < line; k++) {
                    Ed[i][k] = Ed[i][k] ^ E[exist_or_not][k];
                }
                
            }
            else {
                for (int k = 0; k < line; k++) {
                    E[count][k] = Ed[i][k];
                }
                count++;
                break;
            }
        }
    }
    gettimeofday(&end,NULL);
    cout << "special Gauss:" <<((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000 << "ms" << endl;
}

void special_Gauss_NEON(int** E, int** Ed, int row, int rowd, int line) {
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    int count = row - rowd;
    for (int i = 0; i < rowd; i++) {
        while (find1(Ed[i], line) != -1) {
            int exist_or_not = _exist(E, Ed[i], row, line);
            if (exist_or_not != -1) {
                int k;
                for (k = 0; k + 4 <= line; k += 4) {
                    int32x4_t v1 = vld1q_s32((int*)Ed[i] + k);
                    int32x4_t v2 = vld1q_s32((int*)E[exist_or_not] + k);
                    v1 = veorq_s32(v1, v2);
                    vst1q_s32((int*)Ed[i] + k,v1);
                }
                for (; k < line; k ++) {
                    Ed[i][k] = Ed[i][k] ^ E[exist_or_not][k];
                }
                
            }
            else {
                int k;
                for (k = 0; k + 4 <= line; k += 4) {
                    int32x4_t v1 = vld1q_s32((int*)Ed[i] + k);
                    vst1q_s32((int*)E[count] + k,v1);
                }
                for (; k < line; k++) {
                    E[count][k] = Ed[i][k];
                }
                count++;
                break;
            }
        }
    }
    gettimeofday(&end,NULL);
    cout << "NEON:" << ((end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec))*1.0/1000 << "ms" << endl;
}


int main() {
    ifstream xiao;
    ifstream beixiao;
    ofstream result;
    int row = 22;
    int line = 130;
    int rowd = 8;
    int lined = 130;
    row += rowd;
    
    int** E = new int* [row];
    for (int i = 0; i < row; i++)
        E[i] = new int[line];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < line; j++) {
            E[i][j] = 0;
        }
    }
    
    int** Ed = new int* [rowd];
    for (int i = 0; i < rowd; i++)
        Ed[i] = new int[lined];
    for (int i = 0; i < rowd; i++) {
        for (int j = 0; j < lined; j++) {
            Ed[i][j] = 0;
        }
    }
        
    xiao.open("消元子.txt", ios::in);
    if (!xiao.is_open()) {
        cout << "消元子文件打开失败" << endl;
        return 1;
    }

    vector<string> X;
    string xiao1;
    while (getline(xiao, xiao1))
        X.push_back(xiao1);
    xiao.close();
    for (int i = 0; i < X.size(); i++)
        string_to_num(X[i], i, line, E);

    beixiao.open("被消元行.txt", ios::in);
    if (!beixiao.is_open()) {
        cout << "被消元行文件打开失败" << endl;
        return 1;
    }

    vector<string> XB;
    string beixiao1;
    while (getline(beixiao, beixiao1))
        XB.push_back(beixiao1);
    beixiao.close();
    for (int i = 0; i < XB.size(); i++)
        string_to_num(XB[i], i, lined,Ed);
    
    
    special_Gauss(E, Ed, row, rowd, line);
    X.clear();
    XB.clear();


    
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < line; j++) {
            E[i][j] = 0;
        }
    }
    
    
    for (int i = 0; i < rowd; i++) {
        for (int j = 0; j < lined; j++) {
            Ed[i][j] = 0;
        }
    }
        
    xiao.open("消元子.txt", ios::in);
    if (!xiao.is_open()) {
        cout << "消元子文件打开失败" << endl;
        return 1;
    }


    string xiao11;
    while (getline(xiao, xiao11))
        X.push_back(xiao11);
    xiao.close();
    for (int i = 0; i < X.size(); i++)
        string_to_num(X[i], i, line, E);

    beixiao.open("被消元行.txt", ios::in);
    if (!beixiao.is_open()) {
        cout << "被消元行文件打开失败" << endl;
        return 1;
    }


    string beixiao11;
    while (getline(beixiao, beixiao11))
        XB.push_back(beixiao11);
    beixiao.close();
    for (int i = 0; i < XB.size(); i++)
        string_to_num(XB[i], i, lined,Ed);
    
    special_Gauss_NEON(E, Ed, row, rowd, line);


    for (int i = 0; i < row; i++)
        delete[] E[i];
    delete[] E;
    for (int i = 0; i < rowd; i++)
        delete[] Ed[i];
    delete[]Ed;
    
    return 0;
}