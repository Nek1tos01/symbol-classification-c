#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

// Простая матрица double-значений в непрерывном массиве.
typedef struct {
    size_t rows;
    size_t cols;
    double* data;
} Matrix;

// Создаёт матрицу и заполняет её нулями.
Matrix matrix_create(size_t rows, size_t cols);

// Освобождает память матрицы.
void matrix_free(Matrix* m);

// Заполняет матрицу одним значением.
void matrix_fill(Matrix* m, double value);

// Заполняет матрицу случайными значениями.
void matrix_random(Matrix* m, double min, double max);

// Читает элемент матрицы.
double matrix_get(const Matrix* m, size_t r, size_t c);

// Записывает элемент матрицы.
void matrix_set(Matrix* m, size_t r, size_t c, double value);

#endif
