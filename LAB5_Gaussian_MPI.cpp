#include <iostream>
#include <mpi.h>


#define SIZE 2000
#define TOLERANCE 0.000001


double* new_random_matrix() {
    double* matrix = new double[SIZE * (SIZE + 1)];

    for (int i = 0; i < SIZE; i += 1) {
        for (int j = 0; j < SIZE; j += 1) {
            matrix[i * SIZE + j] = rand() % 1000 + 1;
        }

        matrix[i * SIZE + SIZE] = rand() % SIZE;
    }

    return matrix;
}


double* back_steps(double* matrix) {

    double* answer = new double[SIZE];
    for (int i = SIZE - 1; i >= 0; i -= 1) {
        int row_position = i * SIZE;
        answer[i] = matrix[row_position + SIZE];

        for (int j = i + 1; j < SIZE; j += 1) {
            answer[i] -= matrix[row_position + j] * answer[j];
        }

        answer[i] /= matrix[row_position + i];
    }

    return answer;
}


void validate_answer(double* matrix, double* answer) {
    double sum = 0;
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            sum += answer[j] * matrix[i * SIZE + j];
        }
        if (std::abs(sum - matrix[i * SIZE + SIZE]) > TOLERANCE)
            std::cerr << "Неверное решение. Разница = " << sum - matrix[i * SIZE + SIZE] << std::endl;
        sum = 0;
    }
}



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double* matrix = new_random_matrix();
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = SIZE / (size - 1);
    double elements = (float)SIZE / (size - 1);

    if (rows_per_process != elements)
    {
        std::cerr << "Невозможно вычислить с таким количеством процессов" << std::endl;
        return -1;
    }

    if (rank == 0)
    {
        double start_time = MPI_Wtime();
        double* local_a = new double[rows_per_process * (SIZE + 1)];

        int index = 0;
        for (int proc = 1; proc < size; proc++)
        {
            for (int i = 0; i < rows_per_process * (SIZE + 1); i++)
            {
                local_a[i] = matrix[index];
                index++;
            }
            MPI_Send(local_a, rows_per_process * (SIZE + 1), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
        }

        index = 0;

        for (int proc = 1; proc < size; proc++)
        {
            MPI_Recv(local_a, rows_per_process * (SIZE + 1), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < rows_per_process * (SIZE + 1); i++)
            {
                matrix[index] = local_a[i];
                index++;
            }
        }

        
        double* answer = back_steps(matrix);
        double end_time = MPI_Wtime();

        std::cout << "Time = " << end_time - start_time << " s" << std::endl;

        validate_answer(matrix, answer);
        delete[] local_a;
    }
    else
    {
        int rows_start = rows_per_process * (rank - 1);
        int rows_end = rank * rows_per_process - 1;

        double* local_a = new double[rows_per_process * (SIZE + 1)];
        double* local_b = new double[rows_per_process * (SIZE + 1)];

        MPI_Recv(local_a, rows_per_process * (SIZE + 1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        for (int t = 1; t < rank; t += 1)
        {
            MPI_Recv(local_b, rows_per_process * (SIZE + 1), MPI_DOUBLE, t, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

            for (int i = 0; i < rows_per_process; i += 1) {
                for (int j = 0; j < rows_per_process; j += 1) {
                    double coefficient = local_b[i * SIZE + j + rows_per_process * (t - 1)];

                    for (int k = SIZE + 1; k >= j; k -= 1) {
                        local_a[i * SIZE + k] -= coefficient * local_b[j * SIZE + k];
                    }
                }
            }
        }

        delete[] local_b;
        int counter = 0;

        for (int i = 0; i < rows_per_process; i++) {
            int row_position = i * SIZE;
            double coefficient = local_a[row_position + i + rows_start + counter];

            for (int j = (SIZE + 1); j >= rows_start + counter; j -= 1) {
                local_a[row_position + j] /= coefficient;
            }

            for (int j = i + 1; j < rows_per_process; j += 1) {
                coefficient = local_a[j * SIZE + rows_start + counter];

                for (int k = SIZE + 1; k >= rows_start + counter; k -= 1) {
                    local_a[j * SIZE + k] -= coefficient * local_a[i * SIZE + k];
                }
            }
        }

        for (int proc = rank + 1; proc < size; proc++)
        {
            MPI_Send(local_a, rows_per_process * (SIZE + 1), MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
        }

        MPI_Send(local_a, rows_per_process * (SIZE + 1), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        delete[] local_a;
    }

    MPI_Finalize();
    return 0;
}