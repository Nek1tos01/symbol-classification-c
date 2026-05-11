cmake_minimum_required(VERSION 3.10)
project(nn_classifier C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(nn_classifier
    src/main.c
    src/util.c
    src/matrix.c
    src/config.c
    src/data.c
    src/nn.c
)

target_include_directories(nn_classifier PRIVATE include)

target_link_libraries(nn_classifier m)

find_program(PROJECT_PYTHON NAMES py python3 python)

if(PROJECT_PYTHON)
    add_custom_target(visualize
        COMMAND ${PROJECT_PYTHON} "${CMAKE_SOURCE_DIR}/scripts/plot_results.py"
                --input-dir "${CMAKE_SOURCE_DIR}"
                --output-dir "${CMAKE_SOURCE_DIR}/visualization"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Создание Python-визуализации"
    )

    add_custom_target(run_and_visualize
        COMMAND $<TARGET_FILE:nn_classifier> "${CMAKE_SOURCE_DIR}/config.txt"
        COMMAND ${PROJECT_PYTHON} "${CMAKE_SOURCE_DIR}/scripts/plot_results.py"
                --input-dir "${CMAKE_SOURCE_DIR}"
                --output-dir "${CMAKE_SOURCE_DIR}/visualization"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        DEPENDS nn_classifier
        COMMENT "Запуск классификатора и создание Python-визуализации"
    )
else()
    message(STATUS "Python не найден; цели визуализации отключены.")
endif()
