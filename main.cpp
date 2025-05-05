// main.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <omp.h>
#include "mcl.h"

using namespace std;
typedef long long lld;

double** convert_to_double_array(const vector<vector<double>>& matrix, int n) {
    double** arr = new double*[n];
    for (int i = 0; i < n; ++i) {
        arr[i] = new double[n];
        for (int j = 0; j < n; ++j) {
            arr[i][j] = matrix[i][j];
        }
    }
    return arr;
}

bool read_mtx(const string &filename, vector<vector<double>> &matrix, int &n) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return false;
    }

    string line;
    int rows, cols, nnz; 

    while (getline(file, line)) {
        if (line[0] == '%') continue; 

        stringstream ss(line);
        ss >> rows >> cols >> nnz;
        break;  
    }

    n = rows;
    matrix.resize(n, vector<double>(n, 0.0));  

    while (getline(file, line)) {
        if (line[0] == '%') continue;  

        stringstream ss(line);
        int i, j;
        double value;
        ss >> i >> j >> value;

        matrix[i - 1][j - 1] = value;
    }

    file.close();
    std::cout << "  Loaded " << n << "×" << n << " matrix, starting MCL…\n";

    return true;
}

// Deep copy function
double** copy_matrix(double** src, int n) {
    double** dst = new double*[n];
    for (int i = 0; i < n; ++i) {
        dst[i] = new double[n];
        for (int j = 0; j < n; ++j) {
            dst[i][j] = src[i][j];
        }
    }
    return dst;
}


void delete_matrix(double** mat, int n) {
    for (int i = 0; i < n; ++i) {
        delete[] mat[i];
    }
    delete[] mat;
}


int main(int argc, char* argv[]) {
    std::cout << "Loading matrix from: " << argv[1] << std::endl;

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <matrix_file.mtx>\n";
        return 1;
    }
    int n; vector<vector<double>> mat;
    if (!read_mtx(argv[1], mat, n)) return 1;
    double** a = convert_to_double_array(mat, n);

    lld e = 2, r = 2;
    double eps = 1e-6, eps2 = 1e-3;

    double** a_serial = copy_matrix(a, n);
    double** a_parallel = copy_matrix(a, n);

    // PARALLEL run
    double t2 = omp_get_wtime();
    printf("DEBUG: timer before OpenMP MCL:  t2 = %.6f\n", t2);

    auto res_omp = mcl_openmp(a_parallel, n, e, r, eps, eps2);

    double t3 = omp_get_wtime();
    printf("DEBUG: timer after  OpenMP MCL:  t3 = %.6f\n", t3);
    printf("OpenMP MCL: %.6f s, %zu clusters\n", t3 - t2, res_omp.size());

    // SERIAL run
    double t0 = omp_get_wtime();
    auto res_serial = mcl_serial(a_serial, n, e, r, eps, eps2);
    double t1 = omp_get_wtime();
    printf("Serial MCL: %.6f s, %zu clusters\n", t1 - t0, res_serial.size());


    // cleanup
    delete_matrix(a_serial, n);
    delete_matrix(a_parallel, n);
    delete_matrix(a, n);


    // // SERIAL run
    // double t0 = omp_get_wtime();
    // auto res_serial = mcl_serial(a, n, e, r, eps, eps2);
    // double t1 = omp_get_wtime();
    // printf("Serial MCL: %.6f s, %zu clusters\n",
    //        t1 - t0, res_serial.size());

    // // PARALLEL run
    // double t2 = omp_get_wtime();        // <<< reset timer here
    // auto res_omp = mcl_openmp(a, n, e, r, eps, eps2);
    // double t3 = omp_get_wtime();
    // printf("OpenMP MCL: %.6f s, %zu clusters\n",
    //        t3 - t2, res_omp.size());

    // // … print clusters or compare results …

    // // cleanup
    // for (int i = 0; i < n; ++i) delete[] a[i];
    // delete[] a;
    return 0;
}
