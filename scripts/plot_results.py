#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_metrics(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "loss": float(row["loss"]),
                "train_acc": float(row["train_acc"]),
                "val_acc": float(row["val_acc"]),
            })
    return rows


def read_loss_only(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "loss": float(row["loss"]),
                "train_acc": None,
                "val_acc": None,
            })
    return rows


def read_heatmap(path, max_rows=350):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            values = [float(x) for x in line.split()]
            if values:
                rows.append(values)

    if len(rows) <= max_rows:
        return rows

    return [rows[int(i * (len(rows) - 1) / (max_rows - 1))] for i in range(max_rows)]


def write_index(output_dir, metrics_data, heatmap_data):
    epochs = [row["epoch"] for row in metrics_data]
    tick_step = max(1, (max(epochs) - min(epochs)) // 10) if len(epochs) > 1 else 1
    epoch_ticks = [epoch for epoch in epochs if (epoch - epochs[0]) % tick_step == 0]
    if epochs[-1] not in epoch_ticks:
        epoch_ticks.append(epochs[-1])

    metrics_json = json.dumps(metrics_data, ensure_ascii=False)
    heatmap_json = json.dumps(heatmap_data)
    ticks_json = json.dumps(epoch_ticks)

    last = metrics_data[-1]
    has_accuracy = last["train_acc"] is not None and last["val_acc"] is not None
    accuracy_stats = ""
    if has_accuracy:
        accuracy_stats = f'''
                <div class="stat-item">Точность обучения: {last["train_acc"]:.4f}</div>
                <div class="stat-item">Точность проверки: {last["val_acc"]:.4f}</div>'''

    html_content = f'''<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>Визуализация нейросети MNIST</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f1f5f9; color: #1e293b; }}
        .container {{ max-width: 1120px; margin: 36px auto; padding: 0 20px; }}
        header {{ margin-bottom: 24px; border-bottom: 2px solid #e2e8f0; padding-bottom: 18px; }}
        .card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 2px 8px rgb(15 23 42 / 0.08); padding: 22px; margin-bottom: 24px; }}
        h1 {{ margin: 0; font-size: 28px; color: #0f172a; }}
        h2 {{ margin-top: 0; font-size: 20px; color: #334155; }}
        p {{ color: #64748b; }}
        .stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 14px; font-size: 14px; }}
        .stat-item {{ background: #eff6ff; padding: 8px 14px; border-radius: 999px; color: #2563eb; font-weight: 600; }}
        .controls {{ margin-bottom: 12px; }}
        button {{ cursor: pointer; background: #2563eb; color: white; border: none; padding: 8px 14px; border-radius: 6px; font-weight: 600; }}
        button:hover {{ background: #1d4ed8; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Визуализация нейросети MNIST</h1>
            <p>Графики построены по файлу <code>metrics_history.txt</code>, heatmap построена по <code>heatmap.txt</code>.</p>
            <div class="stats">
                <div class="stat-item">Эпох: {len(metrics_data)}</div>
                <div class="stat-item">Первая ошибка: {metrics_data[0]["loss"]:.6f}</div>
                <div class="stat-item">Последняя ошибка: {last["loss"]:.6f}</div>{accuracy_stats}
            </div>
        </header>

        <section class="card">
            <h2>График ошибки Loss</h2>
            <div class="controls">
                <button onclick="toggleLog()">Переключить шкалу: Linear / Log</button>
            </div>
            <div id="lossPlot" style="width:100%;height:460px;"></div>
        </section>

        <section class="card">
            <h2>График точности Accuracy</h2>
            <p>Показывает точность на обучающей и проверочной выборках после каждой эпохи.</p>
            <div id="accuracyPlot" style="width:100%;height:460px;"></div>
        </section>

        <section class="card">
            <h2>Активации скрытого слоя</h2>
            <p>По оси X — нейроны первого скрытого слоя, по оси Y — проверочные примеры.</p>
            <div id="heatPlot" style="width:100%;height:620px;"></div>
        </section>
    </div>

    <script>
        const metricsData = {metrics_json};
        const heatmapData = {heatmap_json};
        const epochTicks = {ticks_json};

        const epochs = metricsData.map(d => d.epoch);
        const losses = metricsData.map(d => d.loss);
        const trainAcc = metricsData.map(d => d.train_acc);
        const valAcc = metricsData.map(d => d.val_acc);

        const commonXAxis = {{
            title: 'Эпоха',
            tickmode: 'array',
            tickvals: epochTicks,
            ticktext: epochTicks.map(String),
            showgrid: true,
            gridcolor: '#e2e8f0',
            zeroline: false
        }};

        let isLog = false;
        Plotly.newPlot('lossPlot', [{{
            x: epochs,
            y: losses,
            type: 'scatter',
            mode: 'lines+markers',
            marker: {{ size: 7, color: '#2563eb' }},
            line: {{ color: '#2563eb', width: 3 }},
            name: 'Ошибка'
        }}], {{
            margin: {{ t: 20, r: 24, b: 70, l: 72 }},
            xaxis: commonXAxis,
            yaxis: {{ title: 'Loss', type: 'linear', gridcolor: '#e2e8f0', zeroline: false }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }}, {{ responsive: true }});

        function toggleLog() {{
            isLog = !isLog;
            Plotly.relayout('lossPlot', {{ 'yaxis.type': isLog ? 'log' : 'linear' }});
        }}

        if (trainAcc.every(v => v !== null) && valAcc.every(v => v !== null)) {{
            Plotly.newPlot('accuracyPlot', [
                {{
                    x: epochs,
                    y: trainAcc,
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: {{ size: 7, color: '#16a34a' }},
                    line: {{ color: '#16a34a', width: 3 }},
                    name: 'Обучение'
                }},
                {{
                    x: epochs,
                    y: valAcc,
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: {{ size: 7, color: '#dc2626' }},
                    line: {{ color: '#dc2626', width: 3 }},
                    name: 'Проверка'
                }}
            ], {{
                margin: {{ t: 20, r: 24, b: 70, l: 72 }},
                xaxis: commonXAxis,
                yaxis: {{ title: 'Accuracy', range: [0, 1.02], tickformat: '.0%', gridcolor: '#e2e8f0', zeroline: false }},
                legend: {{ orientation: 'h', y: 1.12 }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            }}, {{ responsive: true }});
        }} else {{
            document.getElementById('accuracyPlot').innerHTML = '<p>Файл metrics_history.txt не содержит accuracy. Запустите обучение заново.</p>';
        }}

        Plotly.newPlot('heatPlot', [{{
            z: heatmapData,
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: true,
            colorbar: {{ title: 'Активация' }}
        }}], {{
            margin: {{ t: 20, r: 24, b: 70, l: 72 }},
            xaxis: {{ title: 'Индекс нейрона', dtick: 1 }},
            yaxis: {{ title: 'Номер примера' }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        }}, {{ responsive: true }});
    </script>
</body>
</html>
'''
    (output_dir / "index.html").write_text(html_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--output-dir", default="visualization")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = input_dir / "metrics_history.txt"
    loss_path = input_dir / "loss_history.txt"
    heatmap_path = input_dir / "heatmap.txt"

    if metrics_path.exists():
        metrics_data = read_metrics(metrics_path)
    elif loss_path.exists():
        metrics_data = read_loss_only(loss_path)
    else:
        raise SystemExit(f"Ошибка: не найден файл {metrics_path} или {loss_path}")

    if not heatmap_path.exists():
        raise SystemExit(f"Ошибка: не найден файл {heatmap_path}")

    print("Читаю данные обучения...")
    heatmap_data = read_heatmap(heatmap_path)

    print("Генерирую интерактивные графики...")
    write_index(output_dir, metrics_data, heatmap_data)

    print(f"Готово. Откройте файл: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
