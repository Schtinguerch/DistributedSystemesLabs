#include <iostream>
#include <functional>
#include <omp.h>


#define SIZE 2000
#define TOLERANCE 0.000001


bool equals(double a, double b) {
    return abs(a - b) < TOLERANCE;
}


void delete_matrix(double** matrix) {
    for (int i = 0; i < SIZE; i += 1) {
        delete[] matrix[i];
    }

    delete[] matrix;
}


double** generated_random_matrix() {
    double** matrix = new double*[SIZE];

    for (int i = 0; i < SIZE; i += 1) {
        matrix[i] = new double[SIZE + 1];

        for (int j = 0; j < SIZE + 1; j += 1) {
            matrix[i][j] = rand() % 1000;
        }
    }

    return matrix;
}


double measured_time(std::function<double*(double**)> function, double** matrix) {
    clock_t start_time = clock();
    double* solution = function(matrix);

    clock_t end_time = clock();
    delete[] solution;

    return (end_time - start_time) / double(CLOCKS_PER_SEC);
}


double* solve_by_gauss_open_mp(double** matrix) {
    double* solutionVector = new double[SIZE];

    for (int i = 0; i < SIZE - 1; i += 1) {
        #pragma omp parallel for
        for (int j = i + 1; j < SIZE; j += 1) {
            double coefficient = matrix[j][i] / matrix[i][i];

            for (int k = i; k < SIZE + 1; k += 1) {
                matrix[j][k] -= coefficient * matrix[i][k];
            }
        }
    }

    for (int i = SIZE - 1; i >= 0; i -= 1) {
        if (equals(matrix[i][i], 0)) {
            std::cerr << "Неверное решение: " << matrix[i][i] << std::endl;
            return solutionVector;
        }

        for (int j = i + 1; j < SIZE; j += 1) {
            matrix[i][SIZE] -= solutionVector[j] * matrix[i][j];
        }

        solutionVector[i] = matrix[i][SIZE] / matrix[i][i];
    }

    return solutionVector;
}


int main() {
    double** matrix = generated_random_matrix();
    double seconds = measured_time(solve_by_gauss_open_mp, matrix);

    std::cout << "Time = " << seconds << " seconds" << std::endl;
    
    delete_matrix(matrix);
    return 0;
}
