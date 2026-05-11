#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>

// Безопасно выделяет память.
void* xmalloc(size_t size);

// Безопасно выделяет и обнуляет память.
void* xcalloc(size_t count, size_t size);

// Освобождает память.
void xfree(void* ptr);

// Возвращает случайное число из диапазона.
double rand_uniform(double min, double max);

// Инициализирует генератор случайных чисел.
void seed_rng(unsigned int seed);

#endif
