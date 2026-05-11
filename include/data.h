#ifndef DATA_H
#define DATA_H

#include <stddef.h>
#include "matrix.h"
#include "config.h"

// Набор данных: признаки x, one-hot метки y и число примеров.
typedef struct {
    Matrix x;
    Matrix y;
    size_t samples;
} Dataset;

// Создаёт матрицы для заданного числа примеров, признаков и классов.
int dataset_init(Dataset* ds, size_t samples, size_t input_dim, size_t classes);

// Освобождает память набора данных.
void dataset_free(Dataset* ds);

// Загружает данные из CSV/IDX или создаёт синтетику по настройкам.
int load_or_generate_datasets(const Config* cfg, Dataset* train, Dataset* val);

// Добавляет шум к входным пикселям.
void apply_noise(Matrix* x, double fraction, double delta);

#endif
