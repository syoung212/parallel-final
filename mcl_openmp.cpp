#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include "mcl.h"

using namespace std;
typedef long long lld;

// Parallel matrix multiplication with OpenMP
inline double** matrix_multiply(double **a, double **b, int n, int l, int m) {
    double **c = new double*[n];
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        c[i] = new double[m];
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < l; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    return c;
}

// Fast exponentiation (sequential, low cost)
inline double fast_pow(double num, lld pow) {
    double ret = 1.0;
    while (pow) {
        if (pow & 1) ret *= num;
        pow >>= 1;
        num *= num;
    }
    return ret;
}

inline double** expand(double **a, int n, lld e) {
    // Identity matrix
    double **ret = new double*[n];
    for (int i = 0; i < n; ++i) {
        ret[i] = new double[n];
        for (int j = 0; j < n; ++j)
            ret[i][j] = (i == j);
    }
    
    double **base = a;
    while (e) {
        if (e & 1) {
            double **tmp = matrix_multiply(ret, base, n, n, n);
            // free(ret) if needed
            ret = tmp;
        }
        e >>= 1;
        double **tmp2 = matrix_multiply(base, base, n, n, n);
        // free(base) if needed
        base = tmp2;
    }
    return ret;
}

inline void inflate(double **a, int n, lld r, double eps) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        // first pass: exponentiate and accumulate
        for (int j = 0; j < n; j++) {
            double v = a[i][j];
            if (v > eps) {
                double tmp = fast_pow(v, r);
                a[i][j] = (tmp > eps ? tmp : 0.0);
                sum += a[i][j];
            } else {
                a[i][j] = 0.0;
            }
        }
        // second pass: normalization
        if (sum > 0) {
            for (int j = 0; j < n; j++) a[i][j] /= sum;
        }
    }
}

inline double** normalise(double **a, int n, double eps) {
    double **ret = new double*[n];
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        ret[i] = new double[n];
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            if (a[i][j] > eps) {
                ret[i][j] = a[i][j];
                sum += ret[i][j];
            } else {
                ret[i][j] = 0.0;
            }
        }
        if (sum > 0) {
            for (int j = 0; j < n; j++) ret[i][j] /= sum;
        }
    }
    return ret;
}

inline double sq_diff(double **a, double **b, int n) {
    double ret = 0.0;
    #pragma omp parallel for reduction(+:ret) collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double d = a[i][j] - b[i][j];
            ret += d * d;
        }
    }
    return ret;
}

inline vector<int> get_component(int start, vector<vector<int>> &graph, vector<char> &mark) {
    vector<int> comp;
    queue<int> q;
    q.push(start);
    mark[start] = 1;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        comp.push_back(u);
        for (int v : graph[u]) {
            if (!mark[v]) {
                mark[v] = 1;
                q.push(v);
            }
        }
    }
    return comp;
}

inline vector<vector<int>> build_clusters(double **a, int n, double eps) {
    vector<char> mark(n, 0);
    vector<vector<int>> graph(n);
    
    // build adjacency
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (a[i][j] > eps || a[j][i] > eps) {
                #pragma omp critical
                {
                    graph[i].push_back(j);
                    graph[j].push_back(i);
                }
            }
        }
    }
    
    vector<vector<int>> clusters;
    for (int i = 0; i < n; ++i) {
        if (!mark[i]) {
            clusters.push_back(get_component(i, graph, mark));
        }
    }
    return clusters;
}

vector<vector<int>> mcl_openmp(double **a, int n, lld e, lld r, double eps, double eps2) {
    double **m = normalise(a, n, eps);
    double **next_m;
    double diff;              // <<< declare here

    do {
        next_m = expand(m, n, e);
        inflate(next_m, n, r, eps);
        diff = sq_diff(m, next_m, n);  // <<< assign here
        // you might want to free(m) here if you've allocated a fresh matrix
        m = next_m;
    } while (diff > eps2);

    return build_clusters(m, n, eps);
}


