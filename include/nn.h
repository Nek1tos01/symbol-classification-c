#ifndef NN_H
#define NN_H

#include <stddef.h>
#include "matrix.h"
#include "config.h"
#include "data.h"

// Один полносвязный слой: веса и bias.
typedef struct {
    Matrix weights;
    Matrix biases;
} Layer;

// Полная нейросеть: слои, активации и буферы градиентов.
typedef struct {
    Layer* layers;
    size_t num_layers;
    size_t* dims;
    size_t dims_count;

    Matrix* activations;
    Matrix* z_values;

    Matrix* grad_w;
    Matrix* grad_b;
} Network;

// Создаёт сеть по архитектуре из конфига.
int network_init(Network* net, const Config* cfg);

// Освобождает память сети.
void network_free(Network* net);

// Обучает сеть и возвращает последнюю точность на проверочных данных.
double train(Network* net, const Config* cfg, const Dataset* train, const Dataset* val);

// Считает accuracy на наборе данных.
double evaluate_accuracy(Network* net, const Dataset* ds);

// Сохраняет активации скрытого слоя в текстовый файл.
int export_heatmap(Network* net, const Dataset* ds, const char* path);

// Сохраняет активации скрытого слоя как PGM-изображение.
int export_heatmap_pgm(Network* net, const Dataset* ds, const char* path);

// Печатает вероятности классов для нескольких примеров.
void print_sample_probabilities(Network* net, const Dataset* ds, size_t count);

#endif
