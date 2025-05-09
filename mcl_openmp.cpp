#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
#include <cstring>
#include <immintrin.h> // For AVX/SSE instructions
#include "mcl.h"

using namespace std;
typedef long long lld;

// Tile sizes tuned to Perlmutter cache hierarchy
// L1 ~64KB: use inner-block B1=32 -> block footprint ~24KB
// L2 ~512KB: use mid-block B2=128 -> footprint ~384KB
constexpr int B2 = 128;
constexpr int B1 = 32;

// Memory-aligned allocation for better SIMD performance
inline double* aligned_alloc_double(size_t n) {
    void* ptr = nullptr;
    #ifdef _MSC_VER
    ptr = _aligned_malloc(n * sizeof(double), 32); // 32-byte alignment for AVX
    #else
    posix_memalign(&ptr, 32, n * sizeof(double));
    #endif
    return static_cast<double*>(ptr);
}

// Free aligned memory
inline void aligned_free(void* ptr) {
    #ifdef _MSC_VER
    _aligned_free(ptr);
    #else
    free(ptr);
    #endif
}

// Cache-optimized, two-level tiled matrix multiplication
inline double** matrix_multiply(double **a, double **b, int n, int l, int m) {
    // Allocate contiguous memory for result
    double **c = new double*[n];
    double *c_data = aligned_alloc_double((size_t)n * m);
    memset(c_data, 0, (size_t)n * m * sizeof(double));
    for (int i = 0; i < n; i++) c[i] = c_data + (size_t)i * m;

    // Parallel outer tiles
    #pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int ii = 0; ii < n; ii += B2) {
        for (int kk = 0; kk < l; kk += B2) {
            for (int jj = 0; jj < m; jj += B2) {
                int iimax = min(ii + B2, n);
                int kkmax = min(kk + B2, l);
                int jjmax = min(jj + B2, m);

                // Inner tiles (fit L1)
                for (int i = ii; i < iimax; i += B1) {
                    for (int k = kk; k < kkmax; k += B1) {
                        for (int j = jj; j < jjmax; j += B1) {
                            int imax = min(i + B1, iimax);
                            int kmax = min(k + B1, kkmax);
                            int jmax = min(j + B1, jjmax);
                            
                            // Block multiply A[i:imax, k:kmax] * B[k:kmax, j:jmax]
                            for (int ii2 = i; ii2 < imax; ++ii2) {
                                for (int kk2 = k; kk2 < kmax; ++kk2) {
                                    double a_val = a[ii2][kk2];
#ifdef __AVX__
                                    // Vector length 4 doubles = 32 bytes
                                    int jj2 = j;
                                    __m256d a_vec = _mm256_set1_pd(a_val);
                                    for (; jj2 + 4 <= jmax; jj2 += 4) {
                                        __m256d b_vec = _mm256_loadu_pd(&b[kk2][jj2]);
                                        __m256d c_vec = _mm256_loadu_pd(&c[ii2][jj2]);
                                        c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                                        _mm256_storeu_pd(&c[ii2][jj2], c_vec);
                                    }
                                    // Remainder
                                    for (; jj2 < jmax; ++jj2) {
                                        c[ii2][jj2] += a_val * b[kk2][jj2];
                                    }
#else
                                    for (int jj2 = j; jj2 < jmax; ++jj2) {
                                        c[ii2][jj2] += a_val * b[kk2][jj2];
                                    }
#endif
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return c;
}

// Fast exponentiation with bit manipulation - already efficient
inline double fast_pow(double num, lld pow) {
    double ret = 1.0;
    while (pow) {
        if (pow & 1) ret *= num;
        pow >>= 1;
        num *= num;
    }
    return ret;
}

// Free a matrix properly with support for contiguous allocation
inline void free_matrix(double **matrix, int n) {
    if (matrix) {
        if (matrix[0]) {
            // Check if this is a contiguous allocation
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

// Matrix exponentiation with optimized memory management
inline double** expand(double **a, int n, lld e) {
    // Identity matrix with contiguous memory allocation
    double **ret = new double*[n];
    double *ret_data = aligned_alloc_double(n * n);
    memset(ret_data, 0, n * n * sizeof(double)); // Zero-initialize
    
    for (int i = 0; i < n; i++) {
        ret[i] = ret_data + i * n;
        ret[i][i] = 1.0;  // Set diagonal to 1
    }
    
    // Create a copy of input matrix
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

// Sparse matrix inflation optimization - skips zeros and uses SIMD
inline void inflate(double **a, int n, lld r, double eps) {
    // First pass - find non-zero values and apply exponentiation
    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < n; i++) {
        // Count non-zeros to optimize memory access
        int nonzeros = 0;
        for (int j = 0; j < n; j++) {
            if (a[i][j] > eps) nonzeros++;
        }
        
        // Sparse optimization - for rows with few non-zeros
        if (nonzeros < n/4) {  // If row is sparse (< 25% non-zero)
            double sum = 0.0;
            // Process only non-zero elements
            for (int j = 0; j < n; j++) {
                if (a[i][j] > eps) {
                    a[i][j] = fast_pow(a[i][j], r);
                    sum += a[i][j];
                } else {
                    a[i][j] = 0.0;
                }
            }
            
            // Normalization
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
            // Dense row optimization with vectorization
            double sum = 0.0;
            int j = 0;
            
            #ifdef __AVX__
            __m256d eps_vec = _mm256_set1_pd(eps);
            __m256d zeros = _mm256_setzero_pd();
            
            for (; j <= n - 4; j += 4) {
                __m256d values = _mm256_loadu_pd(&a[i][j]);
                __m256d mask = _mm256_cmp_pd(values, eps_vec, _CMP_GT_OQ);
                
                // Calculate powers only for values > eps
                for (int idx = 0; idx < 4; idx++) {
                    double val = ((double*)&values)[idx];
                    if (val > eps) {
                        ((double*)&values)[idx] = fast_pow(val, r);
                        sum += ((double*)&values)[idx];
                    } else {
                        ((double*)&values)[idx] = 0.0;
                    }
                }
                
                // Store results
                _mm256_storeu_pd(&a[i][j], values);
            }
            #endif
            
            // Process remaining elements
            for (; j < n; j++) {
                if (a[i][j] > eps) {
                    a[i][j] = fast_pow(a[i][j], r);
                    sum += a[i][j];
                } else {
                    a[i][j] = 0.0;
                }
            }
            
            // Normalization with vectorization
            if (sum > 0) {
                double inv_sum = 1.0 / sum;
                j = 0;
                
                #ifdef __AVX__
                __m256d inv_sum_vec = _mm256_set1_pd(inv_sum);
                
                for (; j <= n - 4; j += 4) {
                    __m256d values = _mm256_loadu_pd(&a[i][j]);
                    values = _mm256_mul_pd(values, inv_sum_vec);
                    
                    // Set small values to zero
                    __m256d mask = _mm256_cmp_pd(values, eps_vec, _CMP_LE_OQ);
                    values = _mm256_blendv_pd(values, zeros, mask);
                    
                    _mm256_storeu_pd(&a[i][j], values);
                }
                #endif
                
                // Process remaining elements
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

// Optimized normalization with contiguous memory and vectorization
inline double** normalise(double **a, int n, double eps) {
    double **ret = new double*[n];
    double *ret_data = aligned_alloc_double(n * n);
    memset(ret_data, 0, n * n * sizeof(double)); // Zero-initialize
    
    for (int i = 0; i < n; i++) {
        ret[i] = ret_data + i * n;
    }
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        
        // First pass: collect non-zero elements and sum
        for (int j = 0; j < n; j++) {
            if (a[i][j] > eps) {
                ret[i][j] = a[i][j];
                sum += a[i][j];
            }
        }
        
        // Second pass: normalize if sum > 0
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
                result = _mm256_and_pd(result, mask); // Zero out elements that were already 0
                _mm256_storeu_pd(&ret[i][j], result);
            }
            #endif
            
            // Process remaining elements
            for (; j < n; j++) {
                if (ret[i][j] > 0) {
                    ret[i][j] *= inv_sum;
                }
            }
        }
    }
    return ret;
}

// Calculate squared difference between matrices with vectorization
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
            
            // Extract sum from vector
            double partial_sum[4];
            _mm256_storeu_pd(partial_sum, sum_vec);
            local_sum += partial_sum[0] + partial_sum[1] + partial_sum[2] + partial_sum[3];
            #endif
            
            // Process remaining elements
            for (; j < n; j++) {
                double d = a[i][j] - b[i][j];
                local_sum += d * d;
            }
        }
        
        ret += local_sum;
    }
    
    return ret;
}

// Helper function to get connected component (unchanged)
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

// Optimized cluster building with sparse graph representation
inline vector<vector<int>> build_clusters(double **a, int n, double eps) {
    vector<char> mark(n, 0);
    vector<vector<int>> graph(n);
    
    // Count non-zeros to determine sparsity
    int total_edges = 0;
    #pragma omp parallel for reduction(+:total_edges)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j && (a[i][j] > eps || a[j][i] > eps)) {
                total_edges++;
            }
        }
    }
    
    // Pre-allocate memory based on estimated size
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        graph[i].reserve(total_edges / n + 1);
    }
    
    // Build adjacency list with thread-local buffers to reduce contention
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
    
    // Merge thread-local results
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < n; ++i) {
            if (!thread_local_graphs[t][i].empty()) {
                graph[i].insert(graph[i].end(), 
                                thread_local_graphs[t][i].begin(), 
                                thread_local_graphs[t][i].end());
            }
        }
    }
    
    // Find connected components
    vector<vector<int>> clusters;
    for (int i = 0; i < n; ++i) {
        if (!mark[i]) {
            clusters.push_back(get_component(i, graph, mark));
        }
    }
    return clusters;
}

// Main MCL algorithm with optimizations for both small and large matrices
vector<vector<int>> mcl_openmp(double **a, int n, lld e, lld r, double eps, double eps2) {
    // Dynamic thread adjustment based on matrix size and system
    int max_threads = omp_get_max_threads();
    int ideal_threads = min(max_threads, max(1, (n * n) / 10000));
    omp_set_num_threads(ideal_threads);
    
    // Normalize input matrix
    double **m = normalise(a, n, eps);
    double **next_m;
    double diff;
    int iterations = 0;
    const int max_iterations = 100; // Prevent infinite loops
    
    do {
        // Expansion step
        next_m = expand(m, n, e);
        
        // Inflation step
        inflate(next_m, n, r, eps);
        
        // Check convergence
        diff = sq_diff(m, next_m, n);
        
        // Clean up and prepare for next iteration
        free_matrix(m, n);
        m = next_m;
        
        iterations++;
        if (iterations >= max_iterations) break;
        
    } while (diff > eps2);
    
    // Build clusters
    vector<vector<int>> clusters = build_clusters(m, n, eps);
    
    // Clean up
    free_matrix(m, n);
    
    return clusters;
}