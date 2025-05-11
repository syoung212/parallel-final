#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <cstring>
#include <immintrin.h>
#include "mcl.h"

using namespace std;
typedef long long lld;

inline double* aligned_alloc_double(size_t n) {
    void* ptr = nullptr;
    #ifdef _MSC_VER
    ptr = _aligned_malloc(n * sizeof(double), 32); 
    #else
    posix_memalign(&ptr, 32, n * sizeof(double));
    #endif
    return static_cast<double*>(ptr);
}

inline void aligned_free(void* ptr) {
    #ifdef _MSC_VER
    _aligned_free(ptr);
    #else
    free(ptr);
    #endif
}

inline void free_matrix(double **matrix, int n) {
    if (matrix) {
        if (matrix[0]) {
            double* first_row_ptr = matrix[0];
            bool is_contiguous = true;
            for (int i = 1; i < n && is_contiguous; i++) {
                if (matrix[i] != first_row_ptr + i * n) {
                    is_contiguous = false;
                }
            }
            
            if (is_contiguous) {
                aligned_free(matrix[0]);
            } else {
                for (int i = 0; i < n; i++) {
                    delete[] matrix[i];
                }
            }
        }
        delete[] matrix;
    }
}

inline double** matrix_multiply(double **a, double **b, int n, int l, int m) {
    return nullptr;
}

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
    return nullptr;
}

inline void inflate(double **a, int n, lld r, double eps) {}

inline double** normalise(double **a, int n, double eps) {
    double **ret = new double*[n];
    double *ret_data = aligned_alloc_double(n * n);
    memset(ret_data, 0, n * n * sizeof(double)); 
    
    for (int i = 0; i < n; i++) {
        ret[i] = ret_data + i * n;
    }
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        
        for (int j = 0; j < n; j++) {
            if (a[i][j] > eps) {
                ret[i][j] = a[i][j];
                sum += a[i][j];
            }
        }
        
        if (sum > 0) {
            double inv_sum = 1.0 / sum;
            int j = 0;
            
            #ifdef __AVX__
            __m256d inv_sum_vec = _mm256_set1_pd(inv_sum);
            __m256d zeros = _mm256_setzero_pd();
            
            for (; j <= n - 4; j += 4) {
                __m256d values = _mm256_loadu_pd(&ret[i][j]);
                __m256d mask = _mm256_cmp_pd(values, zeros, _CMP_GT_OQ);
                __m256d result = _mm256_mul_pd(values, inv_sum_vec);
                result = _mm256_and_pd(result, mask); 
                _mm256_storeu_pd(&ret[i][j], result);
            }
            #endif
            
            for (; j < n; j++) {
                if (ret[i][j] > 0) {
                    ret[i][j] *= inv_sum;
                }
            }
        }
    }
    return ret;
}

inline double sq_diff(double **a, double **b, int n) {
    double ret = 0.0;
    
    #pragma omp parallel reduction(+:ret)
    {
        double local_sum = 0.0;
        
        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < n; i++) {
            int j = 0;
            
            #ifdef __AVX__
            __m256d sum_vec = _mm256_setzero_pd();
            
            for (; j <= n - 4; j += 4) {
                __m256d a_vec = _mm256_loadu_pd(&a[i][j]);
                __m256d b_vec = _mm256_loadu_pd(&b[i][j]);
                __m256d diff = _mm256_sub_pd(a_vec, b_vec);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(diff, diff));
            }
            
            double partial_sum[4];
            _mm256_storeu_pd(partial_sum, sum_vec);
            local_sum += partial_sum[0] + partial_sum[1] + partial_sum[2] + partial_sum[3];
            #endif
            
            for (; j < n; j++) {
                double d = a[i][j] - b[i][j];
                local_sum += d * d;
            }
        }
        
        ret += local_sum;
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

    int total_edges = 0;
    #pragma omp parallel for reduction(+:total_edges)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j && (a[i][j] > eps || a[j][i] > eps)) {
                total_edges++;
            }
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        graph[i].reserve(total_edges / n + 1);
    }
    
    const int num_threads = omp_get_max_threads();
    vector<vector<vector<int>>> thread_local_graphs(num_threads, vector<vector<int>>(n));
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        
        #pragma omp for
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j && (a[i][j] > eps || a[j][i] > eps)) {
                    thread_local_graphs[thread_id][i].push_back(j);
                }
            }
        }
    }
    
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < n; ++i) {
            if (!thread_local_graphs[t][i].empty()) {
                graph[i].insert(graph[i].end(), 
                                thread_local_graphs[t][i].begin(), 
                                thread_local_graphs[t][i].end());
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

vector<vector<int>> mcl_row(double **a, int n, lld e, lld r, double eps, double eps2) {
    int max_threads = omp_get_max_threads();
    int ideal_threads = min(max_threads, max(1, (n * n) / 10000));
    omp_set_num_threads(ideal_threads);
    
    double **m = normalise(a, n, eps);
    double **next_m;
    double diff;
    int iterations = 0;
    const int max_iterations = 100; 


    std::vector<bool> stable_rows(n, false);
    int stable_count = 0;

    do {
        next_m = new double*[n];
        double* next_m_data = aligned_alloc_double(n * n);
        memset(next_m_data, 0, n * n * sizeof(double)); 
    
        for (int i = 0; i < n; i++) {
            next_m[i] = next_m_data + i * n;
            if (stable_rows[i]) {
                memcpy(next_m[i], m[i], n * sizeof(double));
                continue;
            }
    
            for (int k = 0; k < n; k++) {
                double m_ik = m[i][k];
                if (m_ik <= eps) continue;
    
                for (int j = 0; j < n; j++) {
                    double m_kj = m[k][j];
                    if (m_kj > eps) {
                        next_m[i][j] += m_ik * m_kj;
                    }
                }
            }
        }
    
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < n; i++) {
            if (stable_rows[i]) continue;
    
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (next_m[i][j] > eps) {
                    next_m[i][j] = fast_pow(next_m[i][j], r);
                    sum += next_m[i][j];
                } else {
                    next_m[i][j] = 0.0;
                }
            }
    
            if (sum > 0) {
                double inv_sum = 1.0 / sum;
                for (int j = 0; j < n; j++) {
                    if (next_m[i][j] > 0.0) {
                        next_m[i][j] *= inv_sum;
                        if (next_m[i][j] <= eps) next_m[i][j] = 0.0;
                    }
                }
            }
        }
    
        diff = 0.0;
        #pragma omp parallel for reduction(+:diff) schedule(dynamic, 32)
        for (int i = 0; i < n; i++) {
            if (stable_rows[i]) continue;
    
            double row_diff = 0.0;
            for (int j = 0; j < n; j++) {
                double d = m[i][j] - next_m[i][j];
                row_diff += d * d;
            }
    
            if (row_diff < eps2 / n) {
                #pragma omp critical
                {
                    if (!stable_rows[i]) {
                        stable_rows[i] = true;
                        stable_count++;
                    }
                }
            } else {
                diff += row_diff;
            }
        }
    
        for (int i = 0; i < n; i++) {
            memcpy(m[i], next_m[i], n * sizeof(double));
        }
    
        delete[] next_m;
        free(next_m_data);
    
        iterations++;
        if (iterations >= max_iterations || stable_count == n) break;
    
    } while (diff > eps2);
    
    vector<vector<int>> clusters = build_clusters(m, n, eps);
    
    free_matrix(m, n);
    
    return clusters;
}