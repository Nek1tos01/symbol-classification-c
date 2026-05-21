#include "config.h"
#include "util.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Задаёт безопасные значения по умолчанию, если часть конфига не указана.
static void set_defaults(Config* cfg) {
    memset(cfg, 0, sizeof(*cfg));

    cfg->neurons_count = 3;
    cfg->neurons = (size_t*)xcalloc(cfg->neurons_count, sizeof(size_t));
    cfg->neurons[0] = 64;
    cfg->neurons[1] = 32;
    cfg->neurons[2] = 4;

    cfg->learning_rate = 0.02;
    cfg->regularization = 0.0005;
    cfg->epochs = 200;
    cfg->train_samples = 1200;
    cfg->val_samples = 240;
    cfg->batch_size = 1;
    cfg->noise_fraction = 0.08;
    cfg->noise_delta = 0.2;
    cfg->seed = 42U;
    cfg->data_mode = DATA_MODE_SYNTHETIC;
    cfg->use_csv = 0;

    strncpy(cfg->csv_train_path, "data/train.csv", sizeof(cfg->csv_train_path) - 1);
    strncpy(cfg->csv_val_path, "data/val.csv", sizeof(cfg->csv_val_path) - 1);
    strncpy(cfg->mnist_train_images_path, "data/mnist/train-images-idx3-ubyte", sizeof(cfg->mnist_train_images_path) - 1);
    strncpy(cfg->mnist_train_labels_path, "data/mnist/train-labels-idx1-ubyte", sizeof(cfg->mnist_train_labels_path) - 1);
    strncpy(cfg->mnist_test_images_path, "data/mnist/t10k-images-idx3-ubyte", sizeof(cfg->mnist_test_images_path) - 1);
    strncpy(cfg->mnist_test_labels_path, "data/mnist/t10k-labels-idx1-ubyte", sizeof(cfg->mnist_test_labels_path) - 1);
}

// Убирает пробелы в начале и конце строки, не выделяя новую память.
static char* trim(char* s) {
    char* end;

    while (*s != '\0' && isspace((unsigned char)*s)) {
        ++s;
    }

    if (*s == '\0') {
        return s;
    }

    end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) {
        *end = '\0';
        --end;
    }

    return s;
}

// Разбирает список размеров слоёв из строки вида "784, 64, 10".
static int parse_neurons(const char* value, Config* cfg) {
    char local[512];
    char* token;
    size_t values[64];
    size_t count = 0;

    if (strlen(value) >= sizeof(local)) {
        return -1;
    }

    strncpy(local, value, sizeof(local) - 1);
    local[sizeof(local) - 1] = '\0';

    token = strtok(local, ",");
    while (token != NULL) {
        char* t = trim(token);
        long v = strtol(t, NULL, 10);
        if (v <= 0 || count >= 64) {
            return -1;
        }
        values[count++] = (size_t)v;
        token = strtok(NULL, ",");
    }

    if (count < 2) {
        return -1;
    }

    xfree(cfg->neurons);
    cfg->neurons = (size_t*)xcalloc(count, sizeof(size_t));
    memcpy(cfg->neurons, values, count * sizeof(size_t));
    cfg->neurons_count = count;

    return 0;
}

// Переводит текстовый режим данных в enum DataMode.
static int parse_data_mode(const char* value, DataMode* mode) {
    if (strcmp(value, "synthetic") == 0 || strcmp(value, "SYNTHETIC") == 0 || strcmp(value, "0") == 0) {
        *mode = DATA_MODE_SYNTHETIC;
        return 0;
    }
    if (strcmp(value, "csv") == 0 || strcmp(value, "CSV") == 0 || strcmp(value, "1") == 0) {
        *mode = DATA_MODE_CSV;
        return 0;
    }
    if (strcmp(value, "mnist") == 0 || strcmp(value, "MNIST") == 0 || strcmp(value, "2") == 0) {
        *mode = DATA_MODE_MNIST;
        return 0;
    }
    return -1;
}

// Возвращает человекочитаемое имя режима данных для вывода на экран.
static const char* data_mode_name(DataMode mode) {
    if (mode == DATA_MODE_CSV) {
        return "csv";
    }
    if (mode == DATA_MODE_MNIST) {
        return "mnist";
    }
    return "synthetic";
}

// Загружает config.txt, поддерживая записи key=value и key:value.
// Неизвестные ключи не останавливают программу, но выводятся как предупреждения,
// чтобы конфиг было проще отлаживать.
int load_config(const char* path, Config* cfg) {
    FILE* f;
    char line[1024];
    int line_num = 0;
    int data_mode_explicit = 0;

    set_defaults(cfg);

    f = fopen(path, "r");
    if (f == NULL) {
        fprintf(stderr, "Предупреждение: не удалось открыть %s. Используются значения по умолчанию.\n", path);
        return 0;
    }

    while (fgets(line, sizeof(line), f) != NULL) {
        char* eq;
        char* key;
        char* value;
        char* comment;

        ++line_num;
        key = trim(line);

        if (*key == '\0' || *key == '#') {
            continue;
        }

        comment = strchr(key, '#');
        if (comment != NULL) {
            *comment = '\0';
            key = trim(key);
            if (*key == '\0') {
                continue;
            }
        }

        eq = strchr(key, '=');
        if (eq == NULL) {
            eq = strchr(key, ':');
        }
        if (eq == NULL) {
            fprintf(stderr, "Ошибка конфига в строке %d: ожидается формат key=value или key:value\n", line_num);
            fclose(f);
            return -1;
        }

        *eq = '\0';
        value = trim(eq + 1);
        key = trim(key);

        if (strcmp(key, "neurons") == 0) {
            if (parse_neurons(value, cfg) != 0) {
                fprintf(stderr, "Ошибка конфига: неверный список neurons в строке %d\n", line_num);
                fclose(f);
                return -1;
            }
        } else if (strcmp(key, "learning_rate") == 0) {
            cfg->learning_rate = strtod(value, NULL);
        } else if (strcmp(key, "regularization") == 0) {
            cfg->regularization = strtod(value, NULL);
        } else if (strcmp(key, "epochs") == 0) {
            cfg->epochs = (size_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "train_samples") == 0) {
            cfg->train_samples = (size_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "val_samples") == 0) {
            cfg->val_samples = (size_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "batch_size") == 0) {
            cfg->batch_size = (size_t)strtoul(value, NULL, 10);
        } else if (strcmp(key, "noise_fraction") == 0) {
            cfg->noise_fraction = strtod(value, NULL);
        } else if (strcmp(key, "noise_delta") == 0) {
            cfg->noise_delta = strtod(value, NULL);
        } else if (strcmp(key, "seed") == 0) {
            cfg->seed = (unsigned int)strtoul(value, NULL, 10);
        } else if (strcmp(key, "data_mode") == 0) {
            if (parse_data_mode(value, &cfg->data_mode) != 0) {
                fprintf(stderr, "Ошибка конфига: неверный data_mode '%s' в строке %d\n", value, line_num);
                fclose(f);
                return -1;
            }
            data_mode_explicit = 1;
        } else if (strcmp(key, "use_csv") == 0) {
            cfg->use_csv = (int)strtol(value, NULL, 10);
            if (!data_mode_explicit) {
                cfg->data_mode = cfg->use_csv ? DATA_MODE_CSV : DATA_MODE_SYNTHETIC;
            }
        } else if (strcmp(key, "csv_train_path") == 0) {
            strncpy(cfg->csv_train_path, value, sizeof(cfg->csv_train_path) - 1);
        } else if (strcmp(key, "csv_val_path") == 0) {
            strncpy(cfg->csv_val_path, value, sizeof(cfg->csv_val_path) - 1);
        } else if (strcmp(key, "mnist_train_images_path") == 0) {
            strncpy(cfg->mnist_train_images_path, value, sizeof(cfg->mnist_train_images_path) - 1);
        } else if (strcmp(key, "mnist_train_labels_path") == 0) {
            strncpy(cfg->mnist_train_labels_path, value, sizeof(cfg->mnist_train_labels_path) - 1);
        } else if (strcmp(key, "mnist_test_images_path") == 0) {
            strncpy(cfg->mnist_test_images_path, value, sizeof(cfg->mnist_test_images_path) - 1);
        } else if (strcmp(key, "mnist_test_labels_path") == 0) {
            strncpy(cfg->mnist_test_labels_path, value, sizeof(cfg->mnist_test_labels_path) - 1);
        } else {
            fprintf(stderr, "Предупреждение: неизвестный ключ конфига '%s' в строке %d\n", key, line_num);
        }
    }

    fclose(f);

    if (cfg->learning_rate <= 0.0 || cfg->learning_rate > 1.0) {
        fprintf(stderr, "Ошибка конфига: learning_rate должен быть в диапазоне (0, 1]\n");
        return -1;
    }
    if (cfg->regularization < 0.0) {
        fprintf(stderr, "Ошибка конфига: regularization должен быть >= 0\n");
        return -1;
    }
    if (cfg->epochs == 0 || cfg->train_samples == 0 || cfg->val_samples == 0) {
        fprintf(stderr, "Ошибка конфига: epochs/train_samples/val_samples должны быть > 0\n");
        return -1;
    }
    if (cfg->noise_fraction < 0.0 || cfg->noise_fraction > 1.0) {
        fprintf(stderr, "Ошибка конфига: noise_fraction должен быть в диапазоне [0, 1]\n");
        return -1;
    }
    if (cfg->noise_delta < 0.0) {
        fprintf(stderr, "Ошибка конфига: noise_delta должен быть >= 0\n");
        return -1;
    }

    return 0;
}

// Освобождает динамическую память, выделенную под архитектуру сети.
void free_config(Config* cfg) {
    if (cfg == NULL) {
        return;
    }
    xfree(cfg->neurons);
    cfg->neurons = NULL;
    cfg->neurons_count = 0;
}

// Печатает текущие параметры запуска перед обучением.
void print_config(const Config* cfg) {
    size_t i;

    printf("Конфигурация:\n");
    printf("  нейроны = ");
    for (i = 0; i < cfg->neurons_count; ++i) {
        printf("%zu", cfg->neurons[i]);
        if (i + 1 < cfg->neurons_count) {
            printf(",");
        }
    }
    printf("\n");
    printf("  скорость_обучения = %.6f\n", cfg->learning_rate);
    printf("  регуляризация = %.6f\n", cfg->regularization);
    printf("  эпохи = %zu\n", cfg->epochs);
    printf("  обучающих_примеров = %zu\n", cfg->train_samples);
    printf("  проверочных_примеров = %zu\n", cfg->val_samples);
    printf("  доля_шума = %.3f\n", cfg->noise_fraction);
    printf("  величина_шума = %.3f\n", cfg->noise_delta);
    printf("  режим_данных = %d (%s)\n", (int)cfg->data_mode, data_mode_name(cfg->data_mode));
    printf("  использовать_csv = %d\n", cfg->use_csv);
    if (cfg->data_mode == DATA_MODE_CSV) {
        printf("  csv_train_path = %s\n", cfg->csv_train_path);
        printf("  csv_val_path = %s\n", cfg->csv_val_path);
    } else if (cfg->data_mode == DATA_MODE_MNIST) {
        printf("  mnist_train_images_path = %s\n", cfg->mnist_train_images_path);
        printf("  mnist_train_labels_path = %s\n", cfg->mnist_train_labels_path);
        printf("  mnist_test_images_path = %s\n", cfg->mnist_test_images_path);
        printf("  mnist_test_labels_path = %s\n", cfg->mnist_test_labels_path);
    }
}
