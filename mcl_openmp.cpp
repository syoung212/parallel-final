// THERE ARE TWO VERSIONS OF THIS CODE. 

// VERSION 1: LOOP UNROLLING + SIMD VECTORIZATION

// VERSION 2: TILING 

// TO RUN TILING (BETTER FOR LARGER MATRICES), RUN VERSION 2 (COMMENT OUT VERSION 1). 

// BY DEFAULT, BELOW IS VERSION 1 


// VERSION 1: LOOP UNROLLING + SIMD VECTORIZATION
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


inline double** matrix_multiply(double **a, double **b, int n, int l, int m) {
    double **c = new double*[n];
    double *c_data = aligned_alloc_double(n * m);
    memset(c_data, 0, n * m * sizeof(double));
    for (int i = 0; i < n; i++) {
        c[i] = c_data + i * m;
    }
    
    if (n * l * m < 10000) {
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < l; k++) { 
                double a_ik = a[i][k];
                for (int j = 0; j < m; j++) {
                    c[i][j] += a_ik * b[k][j];
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < l; k++) {
                double a_ik = a[i][k];
                int j = 0;
                #ifdef __AVX__
                __m256d a_vec = _mm256_set1_pd(a_ik);
                for (; j <= m - 64; j += 64) {
                    __m256d b0  = _mm256_loadu_pd(&b[k][j+ 0]);
                    __m256d b1  = _mm256_loadu_pd(&b[k][j+ 4]);
                    __m256d b2  = _mm256_loadu_pd(&b[k][j+ 8]);
                    __m256d b3  = _mm256_loadu_pd(&b[k][j+12]);
                    __m256d b4  = _mm256_loadu_pd(&b[k][j+16]);
                    __m256d b5  = _mm256_loadu_pd(&b[k][j+20]);
                    __m256d b6  = _mm256_loadu_pd(&b[k][j+24]);
                    __m256d b7  = _mm256_loadu_pd(&b[k][j+28]);
                    __m256d b8  = _mm256_loadu_pd(&b[k][j+32]);
                    __m256d b9  = _mm256_loadu_pd(&b[k][j+36]);
                    __m256d b10 = _mm256_loadu_pd(&b[k][j+40]);
                    __m256d b11 = _mm256_loadu_pd(&b[k][j+44]);
                    __m256d b12 = _mm256_loadu_pd(&b[k][j+48]);
                    __m256d b13 = _mm256_loadu_pd(&b[k][j+52]);
                    __m256d b14 = _mm256_loadu_pd(&b[k][j+56]);
                    __m256d b15 = _mm256_loadu_pd(&b[k][j+60]);

                    __m256d c0  = _mm256_loadu_pd(&c[i][j+ 0]);
                    __m256d c1  = _mm256_loadu_pd(&c[i][j+ 4]);
                    __m256d c2  = _mm256_loadu_pd(&c[i][j+ 8]);
                    __m256d c3  = _mm256_loadu_pd(&c[i][j+12]);
                    __m256d c4  = _mm256_loadu_pd(&c[i][j+16]);
                    __m256d c5  = _mm256_loadu_pd(&c[i][j+20]);
                    __m256d c6  = _mm256_loadu_pd(&c[i][j+24]);
                    __m256d c7  = _mm256_loadu_pd(&c[i][j+28]);
                    __m256d c8  = _mm256_loadu_pd(&c[i][j+32]);
                    __m256d c9  = _mm256_loadu_pd(&c[i][j+36]);
                    __m256d c10 = _mm256_loadu_pd(&c[i][j+40]);
                    __m256d c11 = _mm256_loadu_pd(&c[i][j+44]);
                    __m256d c12 = _mm256_loadu_pd(&c[i][j+48]);
                    __m256d c13 = _mm256_loadu_pd(&c[i][j+52]);
                    __m256d c14 = _mm256_loadu_pd(&c[i][j+56]);
                    __m256d c15 = _mm256_loadu_pd(&c[i][j+60]);

                    c0  = _mm256_fmadd_pd(a_vec, b0,  c0);
                    c1  = _mm256_fmadd_pd(a_vec, b1,  c1);
                    c2  = _mm256_fmadd_pd(a_vec, b2,  c2);
                    c3  = _mm256_fmadd_pd(a_vec, b3,  c3);
                    c4  = _mm256_fmadd_pd(a_vec, b4,  c4);
                    c5  = _mm256_fmadd_pd(a_vec, b5,  c5);
                    c6  = _mm256_fmadd_pd(a_vec, b6,  c6);
                    c7  = _mm256_fmadd_pd(a_vec, b7,  c7);
                    c8  = _mm256_fmadd_pd(a_vec, b8,  c8);
                    c9  = _mm256_fmadd_pd(a_vec, b9,  c9);
                    c10 = _mm256_fmadd_pd(a_vec, b10, c10);
                    c11 = _mm256_fmadd_pd(a_vec, b11, c11);
                    c12 = _mm256_fmadd_pd(a_vec, b12, c12);
                    c13 = _mm256_fmadd_pd(a_vec, b13, c13);
                    c14 = _mm256_fmadd_pd(a_vec, b14, c14);
                    c15 = _mm256_fmadd_pd(a_vec, b15, c15);

                    _mm256_storeu_pd(&c[i][j+ 0],  c0);
                    _mm256_storeu_pd(&c[i][j+ 4],  c1);
                    _mm256_storeu_pd(&c[i][j+ 8],  c2);
                    _mm256_storeu_pd(&c[i][j+12],  c3);
                    _mm256_storeu_pd(&c[i][j+16],  c4);
                    _mm256_storeu_pd(&c[i][j+20],  c5);
                    _mm256_storeu_pd(&c[i][j+24],  c6);
                    _mm256_storeu_pd(&c[i][j+28],  c7);
                    _mm256_storeu_pd(&c[i][j+32],  c8);
                    _mm256_storeu_pd(&c[i][j+36],  c9);
                    _mm256_storeu_pd(&c[i][j+40],  c10);
                    _mm256_storeu_pd(&c[i][j+44],  c11);
                    _mm256_storeu_pd(&c[i][j+48],  c12);
                    _mm256_storeu_pd(&c[i][j+52],  c13);
                    _mm256_storeu_pd(&c[i][j+56],  c14);
                    _mm256_storeu_pd(&c[i][j+60],  c15);
                }
                for (; j <= m - 32; j += 32) {
                    __m256d b0 = _mm256_loadu_pd(&b[k][j+ 0]);
                    __m256d b1 = _mm256_loadu_pd(&b[k][j+ 4]);
                    __m256d b2 = _mm256_loadu_pd(&b[k][j+ 8]);
                    __m256d b3 = _mm256_loadu_pd(&b[k][j+12]);
                    __m256d b4 = _mm256_loadu_pd(&b[k][j+16]);
                    __m256d b5 = _mm256_loadu_pd(&b[k][j+20]);
                    __m256d b6 = _mm256_loadu_pd(&b[k][j+24]);
                    __m256d b7 = _mm256_loadu_pd(&b[k][j+28]);

                    __m256d c0 = _mm256_loadu_pd(&c[i][j+ 0]);
                    __m256d c1 = _mm256_loadu_pd(&c[i][j+ 4]);
                    __m256d c2 = _mm256_loadu_pd(&c[i][j+ 8]);
                    __m256d c3 = _mm256_loadu_pd(&c[i][j+12]);
                    __m256d c4 = _mm256_loadu_pd(&c[i][j+16]);
                    __m256d c5 = _mm256_loadu_pd(&c[i][j+20]);
                    __m256d c6 = _mm256_loadu_pd(&c[i][j+24]);
                    __m256d c7 = _mm256_loadu_pd(&c[i][j+28]);

                    c0 = _mm256_fmadd_pd(a_vec, b0, c0);
                    c1 = _mm256_fmadd_pd(a_vec, b1, c1);
                    c2 = _mm256_fmadd_pd(a_vec, b2, c2);
                    c3 = _mm256_fmadd_pd(a_vec, b3, c3);
                    c4 = _mm256_fmadd_pd(a_vec, b4, c4);
                    c5 = _mm256_fmadd_pd(a_vec, b5, c5);
                    c6 = _mm256_fmadd_pd(a_vec, b6, c6);
                    c7 = _mm256_fmadd_pd(a_vec, b7, c7);

                    _mm256_storeu_pd(&c[i][j+ 0], c0);
                    _mm256_storeu_pd(&c[i][j+ 4], c1);
                    _mm256_storeu_pd(&c[i][j+ 8], c2);
                    _mm256_storeu_pd(&c[i][j+12], c3);
                    _mm256_storeu_pd(&c[i][j+16], c4);
                    _mm256_storeu_pd(&c[i][j+20], c5);
                    _mm256_storeu_pd(&c[i][j+24], c6);
                    _mm256_storeu_pd(&c[i][j+28], c7);
                }
                for (; j <= m - 16; j += 16) {
                    __m256d b0 = _mm256_loadu_pd(&b[k][j +  0]);
                    __m256d b1 = _mm256_loadu_pd(&b[k][j +  4]);
                    __m256d b2 = _mm256_loadu_pd(&b[k][j +  8]);
                    __m256d b3 = _mm256_loadu_pd(&b[k][j + 12]);

                    __m256d c0 = _mm256_loadu_pd(&c[i][j +  0]);
                    __m256d c1 = _mm256_loadu_pd(&c[i][j +  4]);
                    __m256d c2 = _mm256_loadu_pd(&c[i][j +  8]);
                    __m256d c3 = _mm256_loadu_pd(&c[i][j + 12]);

                    c0 = _mm256_fmadd_pd(a_vec, b0, c0);
                    c1 = _mm256_fmadd_pd(a_vec, b1, c1);
                    c2 = _mm256_fmadd_pd(a_vec, b2, c2);
                    c3 = _mm256_fmadd_pd(a_vec, b3, c3);

                    _mm256_storeu_pd(&c[i][j +  0], c0);
                    _mm256_storeu_pd(&c[i][j +  4], c1);
                    _mm256_storeu_pd(&c[i][j +  8], c2);
                    _mm256_storeu_pd(&c[i][j + 12], c3);
                }

                for (; j <= m - 8; j += 8) {
                    __m256d b0 = _mm256_loadu_pd(&b[k][j +  0]);
                    __m256d b1 = _mm256_loadu_pd(&b[k][j +  4]);
                    __m256d c0 = _mm256_loadu_pd(&c[i][j +  0]);
                    __m256d c1 = _mm256_loadu_pd(&c[i][j +  4]);

                    c0 = _mm256_fmadd_pd(a_vec, b0, c0);
                    c1 = _mm256_fmadd_pd(a_vec, b1, c1);
                    _mm256_storeu_pd(&c[i][j +  0], c0);
                    _mm256_storeu_pd(&c[i][j +  4], c1);
                }

                for (; j <= m - 4; j += 4) {
                    __m256d b0 = _mm256_loadu_pd(&b[k][j]);
                    __m256d c0 = _mm256_loadu_pd(&c[i][j]);
                    c0 = _mm256_fmadd_pd(a_vec, b0, c0);
                    _mm256_storeu_pd(&c[i][j], c0);
                }
                #endif

                for (; j < m; ++j) {
                    c[i][j] += a_ik * b[k][j];
                }
            }
        }
    }
    return c;
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


inline double** expand(double **a, int n, lld e) {
    double **ret = new double*[n];
    double *ret_data = aligned_alloc_double(n * n);
    memset(ret_data, 0, n * n * sizeof(double)); 
    
    for (int i = 0; i < n; i++) {
        ret[i] = ret_data + i * n;
        ret[i][i] = 1.0; 
    }
    
    double **base = new double*[n];
    double *base_data = aligned_alloc_double(n * n);
    
    for (int i = 0; i < n; i++) {
        base[i] = base_data + i * n;
        memcpy(base[i], a[i], n * sizeof(double));
    }
    
    while (e) {
        if (e & 1) {
            double **tmp = matrix_multiply(ret, base, n, n, n);
            free_matrix(ret, n);
            ret = tmp;
        }
        e >>= 1;
        double **tmp2 = matrix_multiply(base, base, n, n, n);
        free_matrix(base, n);
        base = tmp2;
    }
    
    free_matrix(base, n);
    return ret;
}

inline void inflate(double **a, int n, lld r, double eps) {

    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < n; i++) {
        int nonzeros = 0;
        for (int j = 0; j < n; j++) {
            if (a[i][j] > eps) nonzeros++;
        }
        
        if (nonzeros < n/4) { 
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (a[i][j] > eps) {
                    a[i][j] = fast_pow(a[i][j], r);
                    sum += a[i][j];
                } else {
                    a[i][j] = 0.0;
                }
            }
            
            if (sum > 0) {
                double inv_sum = 1.0 / sum;
                for (int j = 0; j < n; j++) {
                    if (a[i][j] > 0) {
                        a[i][j] *= inv_sum;
                        if (a[i][j] <= eps) a[i][j] = 0.0;
                    }
                }
            }
        } else {
            double sum = 0.0;
            int j = 0;
            
            #ifdef __AVX__
            __m256d eps_vec = _mm256_set1_pd(eps);
            __m256d zeros = _mm256_setzero_pd();
            for (; j <= n - 4; j += 4) {
                __m256d values = _mm256_loadu_pd(&a[i][j]);
                __m256d mask = _mm256_cmp_pd(values, eps_vec, _CMP_GT_OQ);
                for (int idx = 0; idx < 4; idx++) {
                    double val = ((double*)&values)[idx];
                    if (val > eps) {
                        ((double*)&values)[idx] = fast_pow(val, r);
                        sum += ((double*)&values)[idx];
                    } else {
                        ((double*)&values)[idx] = 0.0;
                    }
                }
                _mm256_storeu_pd(&a[i][j], values);
            }
            #endif

            for (; j < n; j++) {
                if (a[i][j] > eps) {
                    a[i][j] = fast_pow(a[i][j], r);
                    sum += a[i][j];
                } else {
                    a[i][j] = 0.0;
                }
            }
            
            if (sum > 0) {
                double inv_sum = 1.0 / sum;
                j = 0;
                
                #ifdef __AVX__
                __m256d inv_sum_vec = _mm256_set1_pd(inv_sum);
                
                for (; j <= n - 4; j += 4) {
                    __m256d values = _mm256_loadu_pd(&a[i][j]);
                    values = _mm256_mul_pd(values, inv_sum_vec);
                    __m256d mask = _mm256_cmp_pd(values, eps_vec, _CMP_LE_OQ);
                    values = _mm256_blendv_pd(values, zeros, mask);
                    _mm256_storeu_pd(&a[i][j], values);
                }
                #endif

                for (; j < n; j++) {
                    if (a[i][j] > 0) {
                        a[i][j] *= inv_sum;
                        if (a[i][j] <= eps) a[i][j] = 0.0;
                    }
                }
            }
        }
    }
}

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

vector<vector<int>> mcl_openmp(double **a, int n, lld e, lld r, double eps, double eps2) {
    int max_threads = omp_get_max_threads();
    int ideal_threads = min(max_threads, max(1, (n * n) / 10000));
    omp_set_num_threads(ideal_threads);

    double **m = normalise(a, n, eps);
    double **next_m;
    double diff;
    int iterations = 0;
    const int max_iterations = 100; 
    
    do {
        next_m = expand(m, n, e);
        inflate(next_m, n, r, eps);
        diff = sq_diff(m, next_m, n);
        free_matrix(m, n);
        m = next_m;
        
        iterations++;
        if (iterations >= max_iterations) break;
        
    } while (diff > eps2);
    
    vector<vector<int>> clusters = build_clusters(m, n, eps);
    free_matrix(m, n);
    
    return clusters;
}


// VERSION 2: LOOP UNROLLING + SIMD VECTORIZATION