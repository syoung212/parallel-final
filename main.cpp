#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include "mcl.cpp" 

using namespace std;

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
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <matrix_file.mtx>" << endl;
        return EXIT_FAILURE;
    }

    string filename = argv[1];

    vector<vector<double>> matrix;
    int n;

    if (!read_mtx(filename, matrix, n)) {
        return EXIT_FAILURE;
    }

    double** matrix_arr = convert_to_double_array(matrix, n);

    lld e = 2;  
    lld r = 2;  
    double eps = 1e-6; 
    double eps2 = 1e-3;  

    clock_t start_time = clock();

    vector<vector<int>> result = mcl(matrix_arr, n, e, r, eps, eps2);

    clock_t end_time = clock();

    double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    cout << "MCL executed in " << elapsed_time << " seconds." << endl;

    cout << "MCL found " << result.size() << " clusters:" << endl;
    for (size_t i = 0; i < result.size(); i++) {
        cout << "{";
        for (size_t j = 0; j < result[i].size(); j++) {
            cout << result[i][j] << (j < result[i].size() - 1 ? ", " : "");
        }
        cout << "}" << endl;
    }

    for (int i = 0; i < n; ++i) {
        delete[] matrix_arr[i];
    }
    delete[] matrix_arr;

    return 0;
}
