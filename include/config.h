#ifndef CONFIG_H
#define CONFIG_H

#include <stddef.h>

typedef enum {
    DATA_MODE_SYNTHETIC = 0,
    DATA_MODE_CSV = 1,
    DATA_MODE_MNIST = 2
} DataMode;

// Все параметры запуска: архитектура сети, обучение, шум и пути к данным.
typedef struct {
    size_t* neurons;
    size_t neurons_count;
    double learning_rate;
    double regularization;
    size_t epochs;
    size_t train_samples;
    size_t val_samples;
    size_t batch_size;
    double noise_fraction;
    double noise_delta;
    unsigned int seed;
    DataMode data_mode;
    int use_csv;
    char csv_train_path[260];
    char csv_val_path[260];
    char mnist_train_images_path[260];
    char mnist_train_labels_path[260];
    char mnist_test_images_path[260];
    char mnist_test_labels_path[260];
} Config;

// Загружает настройки из config.txt.
int load_config(const char* path, Config* cfg);

// Освобождает динамические поля конфига.
void free_config(Config* cfg);

// Печатает параметры запуска.
void print_config(const Config* cfg);

#endif
