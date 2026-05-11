#include "data.h"
#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Ограничивает число заданным диапазоном.
static double clamp(double v, double lo, double hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

// Нормализует пиксель из CSV: поддерживает как 0..1, так и 0..255.
static double normalize_csv_pixel(double v) {
    if (v > 1.0) {
        v /= 255.0;
    }
    return clamp(v, 0.0, 1.0);
}

// Выделяет матрицы признаков и one-hot меток для набора данных.
int dataset_init(Dataset* ds, size_t samples, size_t input_dim, size_t classes) {
    if (samples == 0 || input_dim == 0 || classes == 0) {
        return -1;
    }

    ds->x = matrix_create(samples, input_dim);
    ds->y = matrix_create(samples, classes);
    ds->samples = samples;

    return 0;
}

// Освобождает память набора данных.
void dataset_free(Dataset* ds) {
    if (ds == NULL) {
        return;
    }
    matrix_free(&ds->x);
    matrix_free(&ds->y);
    ds->samples = 0;
}

// Строит простой синтетический паттерн для указанного класса.
static void build_pattern(double* out, size_t input_dim, size_t cls, size_t classes) {
    size_t i;
    size_t width = (size_t)sqrt((double)input_dim);
    if (width == 0) {
        width = 1;
    }

    for (i = 0; i < input_dim; ++i) {
        size_t r = i / width;
        size_t c = i % width;
        double base = 0.05;

        if (classes >= 1 && cls == 0 && c < width / 4 + 1) {
            base = 0.95;
        } else if (classes >= 2 && cls == 1 && r < width / 4 + 1) {
            base = 0.95;
        } else if (classes >= 3 && cls == 2 && (r > c ? r - c : c - r) <= 1) {
            base = 0.95;
        } else if (classes >= 4 && cls == 3 && (r + c >= width - 2 && r + c <= width + 1)) {
            base = 0.95;
        } else if (cls > 3 && ((r + cls) % ((cls % 4) + 2) == 0 || (c + cls) % ((cls % 3) + 2) == 0)) {
            base = 0.85;
        }

        out[i] = base;
    }
}

// Заполняет набор данных синтетическими паттернами и метками.
static void generate_synthetic(Dataset* ds) {
    size_t i;
    size_t input_dim = ds->x.cols;
    size_t classes = ds->y.cols;
    double* pattern = (double*)xcalloc(input_dim, sizeof(double));

    for (i = 0; i < ds->samples; ++i) {
        size_t j;
        size_t cls = i % classes;

        build_pattern(pattern, input_dim, cls, classes);

        for (j = 0; j < input_dim; ++j) {
            double v = pattern[j] + rand_uniform(-0.05, 0.05);
            matrix_set(&ds->x, i, j, clamp(v, 0.0, 1.0));
        }

        for (j = 0; j < classes; ++j) {
            matrix_set(&ds->y, i, j, j == cls ? 1.0 : 0.0);
        }
    }

    xfree(pattern);
}

// Добавляет шум: меняет случайную долю пикселей на +-delta.
void apply_noise(Matrix* x, double fraction, double delta) {
    size_t total;
    size_t changes;
    size_t k;

    if (fraction <= 0.0 || delta <= 0.0) {
        return;
    }

    total = x->rows * x->cols;
    changes = (size_t)((double)total * fraction);

    for (k = 0; k < changes; ++k) {
        size_t idx = (size_t)(rand() % (int)total);
        double sign = rand_uniform(0.0, 1.0) < 0.5 ? -1.0 : 1.0;
        double v = x->data[idx] + sign * delta;
        x->data[idx] = clamp(v, 0.0, 1.0);
    }
}

// Загружает MNIST-подобный CSV: метка класса, затем 784 пикселя.
static int load_csv_file(const char* path, Dataset* ds) {
    FILE* f = fopen(path, "r");
    char line[8192];
    size_t row = 0;

    if (f == NULL) {
        fprintf(stderr, "Ошибка: не удалось открыть CSV '%s'\n", path);
        return -1;
    }

    while (fgets(line, sizeof(line), f) != NULL && row < ds->samples) {
        char* token;
        char* endptr;
        size_t col = 0;
        long label;

        token = strtok(line, ",");
        if (token == NULL) {
            continue;
        }

        label = strtol(token, &endptr, 10);
        if (endptr == token) {
            if (row == 0) {
                continue;
            }
            fprintf(stderr, "Ошибка CSV: неверная метка в строке %zu\n", row);
            fclose(f);
            return -1;
        }
        if (label < 0 || (size_t)label >= ds->y.cols) {
            fprintf(stderr, "Ошибка CSV: неверная метка в строке %zu\n", row);
            fclose(f);
            return -1;
        }

        for (col = 0; col < ds->y.cols; ++col) {
            matrix_set(&ds->y, row, col, col == (size_t)label ? 1.0 : 0.0);
        }

        for (col = 0; col < ds->x.cols; ++col) {
            double v;
            token = strtok(NULL, ",");
            if (token == NULL) {
                fprintf(stderr, "Ошибка CSV: недостаточно признаков в строке %zu\n", row);
                fclose(f);
                return -1;
            }
            v = strtod(token, NULL);
            matrix_set(&ds->x, row, col, normalize_csv_pixel(v));
        }

        ++row;
    }

    fclose(f);

    if (row != ds->samples) {
        fprintf(stderr, "Предупреждение CSV: ожидалось строк: %zu, прочитано: %zu\n", ds->samples, row);
    }

    return 0;
}

// Читает 32-битное число big-endian из IDX-файлов MNIST.
static int read_be_u32(FILE* f, unsigned int* out) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) {
        return -1;
    }
    *out = ((unsigned int)b[0] << 24)
         | ((unsigned int)b[1] << 16)
         | ((unsigned int)b[2] << 8)
         | ((unsigned int)b[3]);
    return 0;
}

// Загружает классический бинарный IDX-формат MNIST, если он используется.
static int load_mnist_idx(const char* images_path, const char* labels_path, Dataset* ds) {
    FILE* fi = fopen(images_path, "rb");
    FILE* fl = fopen(labels_path, "rb");
    unsigned int magic_images;
    unsigned int image_count;
    unsigned int rows;
    unsigned int cols;
    unsigned int magic_labels;
    unsigned int label_count;
    unsigned int pixels_per_image;
    unsigned char* buffer;
    size_t i;

    if (fi == NULL) {
        fprintf(stderr, "Ошибка MNIST: не удалось открыть файл изображений '%s'\n", images_path);
        if (fl != NULL) {
            fclose(fl);
        }
        return -1;
    }
    if (fl == NULL) {
        fprintf(stderr, "Ошибка MNIST: не удалось открыть файл меток '%s'\n", labels_path);
        fclose(fi);
        return -1;
    }

    if (read_be_u32(fi, &magic_images) != 0 || read_be_u32(fi, &image_count) != 0
        || read_be_u32(fi, &rows) != 0 || read_be_u32(fi, &cols) != 0) {
        fprintf(stderr, "Ошибка MNIST: неверный заголовок файла изображений\n");
        fclose(fi);
        fclose(fl);
        return -1;
    }

    if (read_be_u32(fl, &magic_labels) != 0 || read_be_u32(fl, &label_count) != 0) {
        fprintf(stderr, "Ошибка MNIST: неверный заголовок файла меток\n");
        fclose(fi);
        fclose(fl);
        return -1;
    }

    if (magic_images != 2051U || magic_labels != 2049U) {
        fprintf(stderr, "Ошибка MNIST: неверные magic numbers (images=%u labels=%u)\n", magic_images, magic_labels);
        fclose(fi);
        fclose(fl);
        return -1;
    }
    if (image_count != label_count) {
        fprintf(stderr, "Ошибка MNIST: число изображений (%u) не совпадает с числом меток (%u)\n", image_count, label_count);
        fclose(fi);
        fclose(fl);
        return -1;
    }

    pixels_per_image = rows * cols;
    if ((size_t)pixels_per_image != ds->x.cols) {
        fprintf(stderr, "Ошибка MNIST: в конфиге входных нейронов=%zu, а размер изображения=%u x %u (= %u)\n",
                ds->x.cols, rows, cols, pixels_per_image);
        fclose(fi);
        fclose(fl);
        return -1;
    }
    if ((unsigned int)ds->samples > image_count) {
        fprintf(stderr, "Ошибка MNIST: запрошено примеров: %zu, в файле есть: %u\n", ds->samples, image_count);
        fclose(fi);
        fclose(fl);
        return -1;
    }

    buffer = (unsigned char*)xcalloc(pixels_per_image, sizeof(unsigned char));

    for (i = 0; i < ds->samples; ++i) {
        unsigned char label;
        size_t j;

        if (fread(&label, 1, 1, fl) != 1) {
            fprintf(stderr, "Ошибка MNIST: не удалось прочитать метку #%zu\n", i);
            xfree(buffer);
            fclose(fi);
            fclose(fl);
            return -1;
        }
        if ((size_t)label >= ds->y.cols) {
            fprintf(stderr, "Ошибка MNIST: метка %u выходит за диапазон %zu классов\n", (unsigned int)label, ds->y.cols);
            xfree(buffer);
            fclose(fi);
            fclose(fl);
            return -1;
        }

        if (fread(buffer, 1, pixels_per_image, fi) != pixels_per_image) {
            fprintf(stderr, "Ошибка MNIST: не удалось прочитать изображение #%zu\n", i);
            xfree(buffer);
            fclose(fi);
            fclose(fl);
            return -1;
        }

        for (j = 0; j < ds->y.cols; ++j) {
            matrix_set(&ds->y, i, j, j == (size_t)label ? 1.0 : 0.0);
        }
        for (j = 0; j < ds->x.cols; ++j) {
            matrix_set(&ds->x, i, j, (double)buffer[j] / 255.0);
        }
    }

    xfree(buffer);
    fclose(fi);
    fclose(fl);
    return 0;
}

// Выбирает источник данных по конфигу: CSV, IDX MNIST или синтетика.
int load_or_generate_datasets(const Config* cfg, Dataset* train, Dataset* val) {
    size_t input_dim = cfg->neurons[0];
    size_t classes = cfg->neurons[cfg->neurons_count - 1];

    if (dataset_init(train, cfg->train_samples, input_dim, classes) != 0) {
        return -1;
    }
    if (dataset_init(val, cfg->val_samples, input_dim, classes) != 0) {
        dataset_free(train);
        return -1;
    }

    if (cfg->data_mode == DATA_MODE_CSV) {
        if (load_csv_file(cfg->csv_train_path, train) != 0 || load_csv_file(cfg->csv_val_path, val) != 0) {
            dataset_free(train);
            dataset_free(val);
            return -1;
        }
    } else if (cfg->data_mode == DATA_MODE_MNIST) {
        if (load_mnist_idx(cfg->mnist_train_images_path, cfg->mnist_train_labels_path, train) != 0
            || load_mnist_idx(cfg->mnist_test_images_path, cfg->mnist_test_labels_path, val) != 0) {
            dataset_free(train);
            dataset_free(val);
            return -1;
        }
    } else {
        generate_synthetic(train);
        generate_synthetic(val);
    }

    apply_noise(&train->x, cfg->noise_fraction, cfg->noise_delta);

    return 0;
}
