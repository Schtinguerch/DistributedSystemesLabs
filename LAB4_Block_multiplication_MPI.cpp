#include <mpi.h>
#include <iostream>


#define SIZE 2000


void block_multiplication(int* matrix_a, int* matrix_b, int* result_matrix, int block_size) {
	for (int i = 0; i < block_size; i += 1) {
		for (int k = 0; k < block_size; k += 1) {
			for (int j = 0; j < block_size; j += 1) {
				int row_position = i * SIZE;
				result_matrix[row_position + j] = matrix_a[row_position + k] + matrix_b[row_position + j];
			}
		}
	}
}


void mpi_thread_init(int* rank, int* thread_count) {
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, rank);
	MPI_Comm_size(MPI_COMM_WORLD, thread_count);
}


void delete_matrices(int* matrix_a, int* matrix_b, int* out_matrix) {
	delete[] matrix_a;
	delete[] matrix_b;
	delete[] out_matrix;
}

void show_endtime_first_rank(int rank, double start_time) {
	if (rank != 0) return;

	double execution_time = (MPI_Wtime() - start_time);
	std::cout << "Multiplication time = " << execution_time << " seconds" << std::endl;
}


void compute_sizes(int thread_count, int* full_size, int* block_size, int* row_block_size, int* block_count) {
	*full_size = SIZE * SIZE;
	*block_size = *full_size / thread_count;
	*row_block_size = SIZE / thread_count;
	*block_count = *full_size / *block_size;
}


void init_in_first_rank(int rank, int* matrix_a, int* matrix_b, int* out_matrix, double& start_time)
{
	if (rank != 0) return;

	for (int i = 0; i < SIZE; i += 1) {
		for (int j = 0; j < SIZE; j += 1) {
			int row_position = i * SIZE;
			matrix_a[row_position + j] = rand() % 1000;
			matrix_b[row_position + j] = rand() % 1000;
			out_matrix[row_position + j] = 0;
		}
	}

	start_time = MPI_Wtime();
}

int main() {
	int rank, thread_count;
	mpi_thread_init(&rank, &thread_count);

	double start_time;
	int *matrix_a = new int[SIZE * SIZE],
		*matrix_b = new int[SIZE * SIZE],
		*out_matrix = new int[SIZE * SIZE];

	init_in_first_rank(rank, matrix_a, matrix_b, out_matrix, start_time);

	int full_size, block_size, row_block_size, block_count;
	compute_sizes(thread_count, &full_size, &block_size, &row_block_size, &block_count);

	if (block_count != thread_count) {
		std::cerr << "error: different thread and block counts: " << thread_count << " != " << block_count << std::endl;
		delete_matrices(matrix_a, matrix_b, out_matrix);

		MPI_Finalize();
		return 1;
	}

	int root = 0;
	MPI_Bcast(matrix_b, full_size, MPI_INT, root, MPI_COMM_WORLD);

	int* mpi_block_matrix = new int[block_size];
	int* mpi_result_block_matrix = new int[block_size];

	for (int i = 0; i < block_size; i += 1) {
		mpi_result_block_matrix[i] = 0;
	}

	MPI_Scatter(matrix_a, block_size, MPI_INT, mpi_block_matrix, block_size, MPI_INT, root, MPI_COMM_WORLD);
	block_multiplication(mpi_block_matrix, matrix_b, mpi_result_block_matrix, row_block_size);
	MPI_Gather(mpi_result_block_matrix, block_size, MPI_INT, out_matrix, block_size, MPI_INT, root, MPI_COMM_WORLD);

	show_endtime_first_rank(rank, start_time);
	MPI_Finalize();

	delete_matrices(matrix_a, matrix_b, out_matrix);
	delete[] mpi_block_matrix;
	delete[] mpi_result_block_matrix;

	return 0;
}