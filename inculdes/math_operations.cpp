
#include "math_operations.h"

std::vector<double> random_vector(int size){
    std::vector<double> vector(size);
    for(double &e : vector){
        e = ((double)rand()) / RAND_MAX;
    }
    return vector;
}

std::vector<std::vector<double>> random_matrix(int m, int n){
    std::vector<std::vector<double>> matrix(m);
    for(int i = 0; i < m; ++i){
        matrix[i] = random_vector(n);
    }
    return matrix;
}

double averageOfVector(const std::vector<double> &vector){
    double sum = 0;
    for(const double &e : vector){
        sum += e; 
    }
    return sum/vector.size();
}

std::vector<std::vector<double>> matrixScalarMultiplication(const std::vector<std::vector<double>>& matrix, double scalar) {
    std::vector<std::vector<double>> result = matrix;

    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result[i].size(); ++j) {
            result[i][j] *= scalar;
        }
    }

    return result;
}

std::vector<double> vectorScalarMultiplication(const std::vector<double>& vec, double scalar) {
    std::vector<double> result(vec.size());

    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] * scalar;
    }

    return result;
}


// Google Gemini for time reasons
std::vector<double> matrixVectorMultiplication(std::vector<std::vector<double>> matrix, std::vector<double> vector) {
   if (matrix.empty()) {
        if (!vector.empty()) {
             throw std::runtime_error("Matrix is empty, but vector is not. Cannot perform multiplication.");
        }
        return {}; // Leere Matrix * Leerer Vektor = Leerer Vektor
    }

    size_t matrix_cols = matrix[0].size();
    size_t matrix_rows = matrix.size();
    size_t vector_size = vector.size();

    if (matrix_cols != vector_size) {
        throw std::runtime_error("Dimension mismatch: Number of matrix columns must equal vector size.");
    }

    std::vector<double> result_vector(matrix_rows);

    for (size_t i = 0; i < matrix_rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < matrix_cols; ++j) {
            sum += matrix[i][j] * vector[j];
        }
        result_vector[i] = sum; 
    }

    return result_vector;
}

std::vector<double> vectorAddition(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::runtime_error("Dimension mismatch: Vectors must have the same size for addition.");
    }

    std::vector<double> result_vec(vec1.size());

    for (size_t i = 0; i < vec1.size(); ++i) {
        result_vec[i] = vec1[i] + vec2[i];
    }

    return result_vec;
}

std::vector<std::vector<double>> outerProduct(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    std::vector<std::vector<double>> result(vec1.size(), std::vector<double>(vec2.size()));
    for (size_t i = 0; i < vec1.size(); ++i) {
        for (size_t j = 0; j < vec2.size(); ++j) {
            result[i][j] = vec1[i] * vec2[j];
        }
    }
    return result;
}

// ChatGPT
std::vector<double> subtractVectors(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vektoren müssen die gleiche Länge haben.");
    }

    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}


std::vector<double> hadamardProduct(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vektoren müssen die gleiche Länge haben.");
    }

    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) return {};

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows));

    for (size_t i = 0; i < rows; ++i) {
        if (matrix[i].size() != cols) {
            throw std::invalid_argument("Alle Zeilen müssen gleich lang sein.");
        }
        for (size_t j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}


std::vector<std::vector<double>> matrixSubtraction(
    const std::vector<std::vector<double>>& a,
    const std::vector<std::vector<double>>& b
) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Die Matrizen müssen die gleiche Anzahl an Zeilen haben.");
    }

    std::vector<std::vector<double>> result(a.size());

    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) {
            throw std::invalid_argument("Alle Zeilen müssen gleich lang sein.");
        }

        result[i].resize(a[i].size());
        for (size_t j = 0; j < a[i].size(); ++j) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }

    return result;
}