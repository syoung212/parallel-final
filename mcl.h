#pragma once
#include <vector>
typedef long long lld;

std::vector<std::vector<int>> mcl_serial(
    double** a, int n, lld e, lld r, double eps, double eps2);

std::vector<std::vector<int>> mcl_openmp(
    double** a, int n, lld e, lld r, double eps, double eps2);

std::vector<std::vector<int>> mcl_row(
    double** a, int n, lld e, lld r, double eps, double eps2);
