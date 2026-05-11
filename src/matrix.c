#include "matrix.h"
#include "util.h"

#include <stdio.h>

// Создаёт матрицу нужного размера и заполняет её нулями.
Matrix matrix_create(size_t rows, size_t cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (double*)xcalloc(rows * cols, sizeof(double));
    return m;
}

// Освобождает память матрицы и сбрасывает её размеры.
void matrix_free(Matrix* m) {
    if (m == NULL) {
        return;
    }
    xfree(m->data);
    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
}

// Заполняет все элементы матрицы одним значением.
void matrix_fill(Matrix* m, double value) {
    size_t total;
    size_t i;

    if (m == NULL || m->data == NULL) {
        return;
    }

    total = m->rows * m->cols;
    for (i = 0; i < total; ++i) {
        m->data[i] = value;
    }
}

// Заполняет матрицу случайными числами из заданного диапазона.
void matrix_random(Matrix* m, double min, double max) {
    size_t total;
    size_t i;

    if (m == NULL || m->data == NULL) {
        return;
    }

    total = m->rows * m->cols;
    for (i = 0; i < total; ++i) {
        m->data[i] = rand_uniform(min, max);
    }
}

// Возвращает элемент матрицы по строке и столбцу.
double matrix_get(const Matrix* m, size_t r, size_t c) {
    return m->data[r * m->cols + c];
}

// Записывает значение в элемент матрицы по строке и столбцу.
void matrix_set(Matrix* m, size_t r, size_t c, double value) {
    m->data[r * m->cols + c] = value;
}
