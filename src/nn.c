#include "nn.h"
#include "util.h"

#include <math.h>
#include <stdio.h>

// Функция активации ReLU для скрытых слоёв.
static double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

// Производная ReLU для обратного распространения ошибки.
static double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// Ограничивает значение диапазоном; используется при экспорте изображения.
static double clamp_value(double v, double lo, double hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

// Копирует один пример из матрицы данных во входной вектор сети.
static void copy_row_to_vec(const Matrix* src, size_t row, Matrix* dst) {
    size_t j;
    for (j = 0; j < src->cols; ++j) {
        dst->data[j] = matrix_get(src, row, j);
    }
}

// Преобразует логиты в вероятности с защитой от переполнения exp().
static void softmax_stable(Matrix* v) {
    size_t j;
    double max_val = v->data[0];
    double sum = 0.0;

    // Subtract max(logit) before exp to avoid overflow.
    for (j = 1; j < v->cols; ++j) {
        if (v->data[j] > max_val) {
            max_val = v->data[j];
        }
    }

    for (j = 0; j < v->cols; ++j) {
        v->data[j] = exp(v->data[j] - max_val);
        sum += v->data[j];
    }

    if (sum <= 0.0) {
        sum = 1.0;
    }

    for (j = 0; j < v->cols; ++j) {
        v->data[j] /= sum;
    }
}

// Инициализирует веса слоя случайными значениями и нулевыми bias.
static void init_layer(Layer* layer, size_t in_dim, size_t out_dim) {
    double limit;
    layer->weights = matrix_create(out_dim, in_dim);
    layer->biases = matrix_create(1, out_dim);

    limit = sqrt(6.0 / (double)(in_dim + out_dim));
    matrix_random(&layer->weights, -limit, limit);
    matrix_fill(&layer->biases, 0.0);
}

// Создаёт все слои, активации и буферы градиентов по конфигу.
int network_init(Network* net, const Config* cfg) {
    size_t l;

    net->dims_count = cfg->neurons_count;
    net->dims = (size_t*)xcalloc(net->dims_count, sizeof(size_t));
    for (l = 0; l < net->dims_count; ++l) {
        net->dims[l] = cfg->neurons[l];
    }

    net->num_layers = net->dims_count - 1;
    net->layers = (Layer*)xcalloc(net->num_layers, sizeof(Layer));
    net->activations = (Matrix*)xcalloc(net->dims_count, sizeof(Matrix));
    net->z_values = (Matrix*)xcalloc(net->num_layers, sizeof(Matrix));
    net->grad_w = (Matrix*)xcalloc(net->num_layers, sizeof(Matrix));
    net->grad_b = (Matrix*)xcalloc(net->num_layers, sizeof(Matrix));

    for (l = 0; l < net->num_layers; ++l) {
        init_layer(&net->layers[l], net->dims[l], net->dims[l + 1]);
        net->z_values[l] = matrix_create(1, net->dims[l + 1]);
        net->grad_w[l] = matrix_create(net->dims[l + 1], net->dims[l]);
        net->grad_b[l] = matrix_create(1, net->dims[l + 1]);
    }

    for (l = 0; l < net->dims_count; ++l) {
        net->activations[l] = matrix_create(1, net->dims[l]);
    }

    return 0;
}

// Освобождает всю динамическую память нейросети.
void network_free(Network* net) {
    size_t l;

    if (net == NULL) {
        return;
    }

    for (l = 0; l < net->num_layers; ++l) {
        matrix_free(&net->layers[l].weights);
        matrix_free(&net->layers[l].biases);
        matrix_free(&net->z_values[l]);
        matrix_free(&net->grad_w[l]);
        matrix_free(&net->grad_b[l]);
    }

    for (l = 0; l < net->dims_count; ++l) {
        matrix_free(&net->activations[l]);
    }

    xfree(net->layers);
    xfree(net->dims);
    xfree(net->activations);
    xfree(net->z_values);
    xfree(net->grad_w);
    xfree(net->grad_b);

    net->layers = NULL;
    net->dims = NULL;
    net->activations = NULL;
    net->z_values = NULL;
    net->grad_w = NULL;
    net->grad_b = NULL;
    net->num_layers = 0;
    net->dims_count = 0;
}

// Выполняет прямой проход для одного примера.
static void forward_sample(Network* net, const Matrix* x, size_t sample_idx) {
    size_t l;

    copy_row_to_vec(x, sample_idx, &net->activations[0]);

    for (l = 0; l < net->num_layers; ++l) {
        size_t i;
        size_t j;
        Layer* layer = &net->layers[l];
        Matrix* a_prev = &net->activations[l];
        Matrix* z = &net->z_values[l];
        Matrix* a = &net->activations[l + 1];

        for (i = 0; i < layer->weights.rows; ++i) {
            double sum = layer->biases.data[i];
            for (j = 0; j < layer->weights.cols; ++j) {
                sum += matrix_get(&layer->weights, i, j) * a_prev->data[j];
            }
            z->data[i] = sum;
        }

        if (l + 1 == net->dims_count - 1) {
            for (i = 0; i < a->cols; ++i) {
                a->data[i] = z->data[i];
            }
            softmax_stable(a);
        } else {
            for (i = 0; i < a->cols; ++i) {
                a->data[i] = relu(z->data[i]);
            }
        }
    }
}

// Считает cross-entropy loss для одного примера.
static double cross_entropy_one(const Matrix* probs, const Matrix* y, size_t sample_idx) {
    size_t k;
    double loss = 0.0;
    const double eps = 1e-12;

    for (k = 0; k < probs->cols; ++k) {
        double yk = matrix_get(y, sample_idx, k);
        if (yk > 0.0) {
            double p = probs->data[k];
            if (p < eps) {
                p = eps;
            }
            loss -= log(p);
        }
    }

    return loss;
}

// Считает L2-штраф по всем весам сети.
static double l2_penalty(Network* net) {
    size_t l;
    double s = 0.0;

    for (l = 0; l < net->num_layers; ++l) {
        size_t i;
        size_t total = net->layers[l].weights.rows * net->layers[l].weights.cols;
        for (i = 0; i < total; ++i) {
            double w = net->layers[l].weights.data[i];
            s += w * w;
        }
    }

    return 0.5 * s;
}

// Вычисляет ошибки слоёв для одного примера методом backpropagation.
static void backprop_sample(Network* net, const Matrix* y, size_t sample_idx, Matrix* deltas) {
    size_t l;

    // For softmax + cross-entropy: dL/dz = y_hat - y.
    l = net->num_layers - 1;
    {
        size_t k;
        for (k = 0; k < deltas[l].cols; ++k) {
            deltas[l].data[k] = net->activations[l + 1].data[k] - matrix_get(y, sample_idx, k);
        }
    }

    while (l > 0) {
        size_t j;
        size_t i;
        size_t prev = l - 1;

        matrix_fill(&deltas[prev], 0.0);
        for (j = 0; j < net->dims[l]; ++j) {
            double grad = 0.0;
            for (i = 0; i < net->dims[l + 1]; ++i) {
                grad += matrix_get(&net->layers[l].weights, i, j) * deltas[l].data[i];
            }
            // Chain rule through ReLU on hidden layers.
            deltas[prev].data[j] = grad * relu_derivative(net->z_values[prev].data[j]);
        }

        l = prev;
    }
}

// По ошибкам слоёв считает градиенты весов и bias.
static void compute_grads(Network* net, Matrix* deltas) {
    size_t l;

    for (l = 0; l < net->num_layers; ++l) {
        size_t i;
        size_t j;
        Matrix* gw = &net->grad_w[l];
        Matrix* gb = &net->grad_b[l];
        Matrix* a_prev = &net->activations[l];

        for (i = 0; i < gw->rows; ++i) {
            gb->data[i] = deltas[l].data[i];
            for (j = 0; j < gw->cols; ++j) {
                matrix_set(gw, i, j, deltas[l].data[i] * a_prev->data[j]);
            }
        }
    }
}

// Обновляет параметры сети одним шагом SGD с L2-регуляризацией.
static void sgd_step(Network* net, double lr, double reg) {
    size_t l;

    for (l = 0; l < net->num_layers; ++l) {
        size_t i;
        size_t j;

        for (i = 0; i < net->layers[l].weights.rows; ++i) {
            for (j = 0; j < net->layers[l].weights.cols; ++j) {
                double w = matrix_get(&net->layers[l].weights, i, j);
                double g = matrix_get(&net->grad_w[l], i, j);
                // L2 contributes reg * w to gradient.
                w -= lr * (g + reg * w);
                matrix_set(&net->layers[l].weights, i, j, w);
            }
            net->layers[l].biases.data[i] -= lr * net->grad_b[l].data[i];
        }
    }
}

// Возвращает индекс максимального значения в векторе вероятностей.
static size_t argmax(const Matrix* v) {
    size_t i;
    size_t best = 0;
    double best_val = v->data[0];

    for (i = 1; i < v->cols; ++i) {
        if (v->data[i] > best_val) {
            best = i;
            best_val = v->data[i];
        }
    }

    return best;
}

// Считает accuracy сети на заданном наборе данных.
double evaluate_accuracy(Network* net, const Dataset* ds) {
    size_t i;
    size_t correct = 0;

    for (i = 0; i < ds->samples; ++i) {
        size_t pred;
        size_t label;

        forward_sample(net, &ds->x, i);
        pred = argmax(&net->activations[net->dims_count - 1]);

        {
            const Matrix* yrow = &ds->y;
            size_t k;
            label = 0;
            for (k = 0; k < yrow->cols; ++k) {
                if (matrix_get(yrow, i, k) > 0.5) {
                    label = k;
                    break;
                }
            }
        }

        if (pred == label) {
            ++correct;
        }
    }

    return (double)correct / (double)ds->samples;
}

// Основной цикл обучения: SGD по примерам, логирование loss и accuracy.
double train(Network* net, const Config* cfg, const Dataset* train, const Dataset* val) {
    size_t epoch;
    Matrix* deltas = (Matrix*)xcalloc(net->num_layers, sizeof(Matrix));
    double last_val_acc = 0.0;
    FILE* history = fopen("loss_history.txt", "w");

    if (history != NULL) {
        fprintf(history, "epoch,loss\n");
    }

    for (epoch = 0; epoch < net->num_layers; ++epoch) {
        deltas[epoch] = matrix_create(1, net->dims[epoch + 1]);
    }

    for (epoch = 1; epoch <= cfg->epochs; ++epoch) {
        size_t i;
        double loss = 0.0;

        for (i = 0; i < train->samples; ++i) {
            forward_sample(net, &train->x, i);
            loss += cross_entropy_one(&net->activations[net->dims_count - 1], &train->y, i);
            backprop_sample(net, &train->y, i, deltas);
            compute_grads(net, deltas);
            sgd_step(net, cfg->learning_rate, cfg->regularization);
        }

        loss /= (double)train->samples;
        loss += cfg->regularization * l2_penalty(net) / (double)train->samples;
        if (history != NULL) {
            fprintf(history, "%zu,%.6f\n", epoch, loss);
        }

        if (epoch % 10 == 0 || epoch == 1 || epoch == cfg->epochs) {
            double train_acc = evaluate_accuracy(net, train);
            last_val_acc = evaluate_accuracy(net, val);
            printf("эпоха=%zu ошибка=%.6f точность_обучения=%.4f точность_проверки=%.4f\n",
                   epoch, loss, train_acc, last_val_acc);
        }
    }

    for (epoch = 0; epoch < net->num_layers; ++epoch) {
        matrix_free(&deltas[epoch]);
    }
    xfree(deltas);
    if (history != NULL) {
        fclose(history);
    }

    return last_val_acc;
}

// Сохраняет активации первого скрытого слоя в текстовый heatmap.txt.
int export_heatmap(Network* net, const Dataset* ds, const char* path) {
    FILE* f;
    size_t i;

    if (net->num_layers < 1) {
        return -1;
    }

    f = fopen(path, "w");
    if (f == NULL) {
        return -1;
    }

    for (i = 0; i < ds->samples; ++i) {
        size_t j;
        forward_sample(net, &ds->x, i);
        for (j = 0; j < net->activations[1].cols; ++j) {
            fprintf(f, "%.6f", net->activations[1].data[j]);
            if (j + 1 < net->activations[1].cols) {
                fputc(' ', f);
            }
        }
        fputc('\n', f);
    }

    fclose(f);
    return 0;
}

// Сохраняет те же активации как grayscale-изображение PGM.
int export_heatmap_pgm(Network* net, const Dataset* ds, const char* path) {
    FILE* f;
    size_t i;
    size_t width;
    size_t height;
    double min_val = 0.0;
    double max_val = 0.0;
    int has_value = 0;

    if (net->num_layers < 1 || ds->samples == 0) {
        return -1;
    }

    width = net->activations[1].cols;
    height = ds->samples;

    for (i = 0; i < height; ++i) {
        size_t j;
        forward_sample(net, &ds->x, i);
        for (j = 0; j < width; ++j) {
            double v = net->activations[1].data[j];
            if (!has_value || v < min_val) {
                min_val = v;
            }
            if (!has_value || v > max_val) {
                max_val = v;
            }
            has_value = 1;
        }
    }

    f = fopen(path, "wb");
    if (f == NULL) {
        return -1;
    }

    fprintf(f, "P5\n%zu %zu\n255\n", width, height);
    for (i = 0; i < height; ++i) {
        size_t j;
        forward_sample(net, &ds->x, i);
        for (j = 0; j < width; ++j) {
            double scaled = 0.0;
            unsigned char pixel;
            if (max_val > min_val) {
                scaled = (net->activations[1].data[j] - min_val) / (max_val - min_val);
            }
            pixel = (unsigned char)(clamp_value(scaled, 0.0, 1.0) * 255.0);
            fwrite(&pixel, 1, 1, f);
        }
    }

    fclose(f);
    return 0;
}

// Печатает вероятности классов для первых count проверочных примеров.
void print_sample_probabilities(Network* net, const Dataset* ds, size_t count) {
    size_t i;
    size_t n = count < ds->samples ? count : ds->samples;

    printf("\nПримеры предсказаний:\n");
    for (i = 0; i < n; ++i) {
        size_t k;
        forward_sample(net, &ds->x, i);
        printf("пример %zu: ", i);
        for (k = 0; k < ds->y.cols; ++k) {
            printf("c%zu=%.4f ", k, net->activations[net->dims_count - 1].data[k]);
        }
        printf("\n");
    }
}
