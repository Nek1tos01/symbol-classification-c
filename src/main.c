#include "config.h"
#include "data.h"
#include "nn.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>

// Точка входа: читает конфиг, обучает сеть и сохраняет результаты.
int main(int argc, char** argv) {
    const char* config_path = "config.txt";
    Config cfg;
    Dataset train_ds;
    Dataset val;
    Network net;
    double val_acc;

    if (argc > 1) {
        config_path = argv[1];
    }

    if (load_config(config_path, &cfg) != 0) {
        fprintf(stderr, "Не удалось прочитать конфиг: %s\n", config_path);
        return EXIT_FAILURE;
    }

    seed_rng(cfg.seed);
    print_config(&cfg);

    if (load_or_generate_datasets(&cfg, &train_ds, &val) != 0) {
        fprintf(stderr, "Не удалось загрузить или создать набор данных\n");
        free_config(&cfg);
        return EXIT_FAILURE;
    }

    if (network_init(&net, &cfg) != 0) {
        fprintf(stderr, "Не удалось инициализировать нейросеть\n");
        dataset_free(&train_ds);
        dataset_free(&val);
        free_config(&cfg);
        return EXIT_FAILURE;
    }

    val_acc = train(&net, &cfg, &train_ds, &val);
    printf("\nИтоговая точность на проверочных данных: %.4f\n", val_acc);

    if (export_heatmap(&net, &val, "heatmap.txt") != 0) {
        fprintf(stderr, "Предупреждение: не удалось экспортировать heatmap.txt\n");
    } else {
        printf("Активации скрытого слоя экспортированы в heatmap.txt\n");
    }
    if (export_heatmap_pgm(&net, &val, "heatmap.pgm") != 0) {
        fprintf(stderr, "Предупреждение: не удалось экспортировать heatmap.pgm\n");
    } else {
        printf("Изображение активаций экспортировано в heatmap.pgm\n");
    }

    print_sample_probabilities(&net, &val, 5);

    network_free(&net);
    dataset_free(&train_ds);
    dataset_free(&val);
    free_config(&cfg);

    return EXIT_SUCCESS;
}
