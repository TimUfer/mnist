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
std::vector<double> subtractVectors(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> hadamardProduct(const std::vector<double>& a, const std::vector<double>& b);
std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>>& matrix);
std::vector<std::vector<double>> outerProduct(const std::vector<double>& vec1, const std::vector<double>& vec2);
std::vector<std::vector<double>> matrixScalarMultiplication(const std::vector<std::vector<double>>& matrix, double scalar);
std::vector<double> vectorScalarMultiplication(const std::vector<double>& vec, double scalar);
std::vector<std::vector<double>> matrixSubtraction(
    const std::vector<std::vector<double>>& a,
    const std::vector<std::vector<double>>& b
);
#endif 