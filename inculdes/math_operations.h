#ifndef MATH_OPERATIONS_H 
#define MATH_OPERATIONS_H

#include <vector>     
#include <stdexcept>  
#include <bits/stdc++.h>



std::vector<double> matrixVectorMultiplication(std::vector<std::vector<double>> matrix, std::vector<double> vector);
std::vector<double> vectorAddition(const std::vector<double>& vec1, const std::vector<double>& vec2);
std::vector<std::vector<double>> random_matrix(int m, int n);
std::vector<double> random_vector(int size);
double averageOfVector(const std::vector<double> &vector);
#endif 