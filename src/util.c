#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Безопасный malloc: завершает программу, если память не выделена.
void* xmalloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Критическая ошибка: malloc не смог выделить %zu байт\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Безопасный calloc для массивов и матриц проекта.
void* xcalloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    if (ptr == NULL) {
        fprintf(stderr, "Критическая ошибка: calloc не смог выделить %zu x %zu байт\n", count, size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// Обёртка над free для единообразия работы с памятью.
void xfree(void* ptr) {
    free(ptr);
}

// Инициализирует генератор случайных чисел. seed=0 берёт текущее время.
void seed_rng(unsigned int seed) {
    if (seed == 0U) {
        seed = (unsigned int)time(NULL);
    }
    srand(seed);
}

// Возвращает случайное вещественное число из диапазона [min, max].
double rand_uniform(double min, double max) {
    double t = (double)rand() / (double)RAND_MAX;
    return min + t * (max - min);
}
