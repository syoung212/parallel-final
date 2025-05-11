#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <omp.h>
#include <cmath>
#include <map>
#include <set>
#include <algorithm>
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

vector<int> extract_clusters(const vector<vector<int>>& res) {
    vector<int> clusters;
    for (const auto& row : res) {
        clusters.push_back(row[0]);
    }
    return clusters;
}

bool check_correctness(const vector<int>& serial_clusters, const vector<int>& parallel_clusters) {
    if (serial_clusters.size() != parallel_clusters.size()) {
        cerr << "Error: Cluster sizes do not match!" << endl;
        return false;
    }

    bool all_correct = true;

    for (size_t i = 0; i < serial_clusters.size(); ++i) {
        if (serial_clusters[i] != parallel_clusters[i]) {
            all_correct = false;
            cout << "Cluster " << i << " mismatches: Serial: " << serial_clusters[i] << ", Parallel: " << parallel_clusters[i] << endl;
        }
    }

    return all_correct;
}

void save_timing_results(const vector<int>& sizes, const vector<double>& serial_times, const vector<double>& omp_times, const vector<double>& row_times) {
    ofstream out("timing_results.csv");
    out << "Matrix Size,Serial Time,OpenMP Time,Row-parallel Time\n";

    for (size_t i = 0; i < sizes.size(); ++i) {
        out << sizes[i] << "," << serial_times[i] << "," << omp_times[i] << "," << row_times[i] << "\n";
    }

    out.close();
    cout << "Timing results saved to timing_results.csv" << endl;
}

int main(int argc, char* argv[]) {
    vector<int> sizes;
    vector<double> serial_times, omp_times, row_times;

    vector<int> matrix_sizes = {250, 500, 750, 1000, 2000, 5000}; 

    for (int matrix_size : matrix_sizes) {
        stringstream filename;
        filename << "./data/matrix_" << matrix_size << ".mtx"; 
        string matrix_file = filename.str();

        int n;
        vector<vector<double>> mat;
        if (!read_mtx(matrix_file, mat, n)) continue; 
        double** a = convert_to_double_array(mat, n);

        lld e = 2, r = 2;
        double eps = 1e-6, eps2 = 1e-3;

        double** a_serial = copy_matrix(a, n);
        double** a_parallel = copy_matrix(a, n);
        double** a_row = copy_matrix(a, n);

        // SERIAL run
        double t0 = omp_get_wtime();
        vector<vector<int>> res_serial;
        // auto res_serial = mcl_serial(a_serial, n, e, r, eps, eps2);
        double t1 = omp_get_wtime();
        serial_times.push_back(t1 - t0);
         std::cout << "Baseline Serial Time: " << t1 - t0 << " seconds" << std::endl;

        // ROW-PARALLEL run
        double t4 = omp_get_wtime();
        auto res_row = mcl_row(a_row, n, e, r, eps, eps2);
        double t5 = omp_get_wtime();
        row_times.push_back(t5 - t4);
        std::cout << "Row Pruning Optimized Time: " << t5 - t4 << " seconds" << std::endl;

        // PARALLEL run
        double t2 = omp_get_wtime();
        auto res_omp = mcl_openmp(a_parallel, n, e, r, eps, eps2);
        double t3 = omp_get_wtime();
        omp_times.push_back(t3 - t2);
        std::cout << "Full Matrix Optimized Time: " << t3 - t2 << " seconds" << std::endl;

        // // ROW-PARALLEL run
        // double t4 = omp_get_wtime();
        // auto res_row = mcl_row(a_row, n, e, r, eps, eps2);
        // double t5 = omp_get_wtime();
        // row_times.push_back(t5 - t4);
        // std::cout << "Row Pruning Optimized Time: " << t5 - t4 << " seconds" << std::endl;

        sizes.push_back(matrix_size);

        vector<int> serial_clusters = extract_clusters(res_serial);
        vector<int> omp_clusters = extract_clusters(res_omp);
        vector<int> row_clusters = extract_clusters(res_row);

        set<int> serial_set(serial_clusters.begin(), serial_clusters.end());
        set<int> omp_set(omp_clusters.begin(), omp_clusters.end());
        set<int> row_set(row_clusters.begin(), row_clusters.end());

        cout << "Matrix size: " << matrix_size << endl;
        cout << "  Serial clusters:     " << serial_set.size() << endl;
        cout << "  Full Matrix Optimized clusters:     " << omp_set.size() << endl;
        cout << "  Row Pruning Optimized clusters: " << row_set.size() << endl;


        delete_matrix(a_serial, n);
        delete_matrix(a_parallel, n);
        delete_matrix(a_row, n);
        delete_matrix(a, n);
    }

    save_timing_results(sizes, serial_times, omp_times, row_times);

    return 0;
}
