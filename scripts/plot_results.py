#!/usr/bin/env python3
import argparse
import csv
import html
from pathlib import Path


def read_loss(path):
    points = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append((int(row["epoch"]), float(row["loss"])))
    return points


def read_heatmap(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            values = [float(x) for x in line.split()]
            if values:
                rows.append(values)
    return rows


def scale(value, src_min, src_max, dst_min, dst_max):
    if src_max <= src_min:
        return (dst_min + dst_max) / 2.0
    t = (value - src_min) / (src_max - src_min)
    return dst_min + t * (dst_max - dst_min)


def write_loss_svg(points, path):
    width = 920
    height = 520
    left = 72
    right = 28
    top = 36
    bottom = 64
    plot_w = width - left - right
    plot_h = height - top - bottom

    epochs = [p[0] for p in points]
    losses = [p[1] for p in points]
    x_min, x_max = min(epochs), max(epochs)
    y_min, y_max = 0.0, max(losses)

    coords = []
    for epoch, loss in points:
        x = scale(epoch, x_min, x_max, left, left + plot_w)
        y = scale(loss, y_min, y_max, top + plot_h, top)
        coords.append(f"{x:.2f},{y:.2f}")

    grid = []
    for i in range(6):
        y = top + plot_h * i / 5.0
        value = y_max * (5 - i) / 5.0
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#475569">{value:.3f}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="24" font-size="20" font-family="Arial, sans-serif" fill="#111827">Ошибка cross-entropy по эпохам</text>
{''.join(grid)}
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.5"/>
<polyline points="{' '.join(coords)}" fill="none" stroke="#2563eb" stroke-width="3"/>
<circle cx="{coords[-1].split(',')[0]}" cy="{coords[-1].split(',')[1]}" r="4" fill="#dc2626"/>
<text x="{left + plot_w / 2:.2f}" y="{height - 20}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#334155">эпоха</text>
<text x="18" y="{top + plot_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2:.2f})" font-size="14" font-family="Arial, sans-serif" fill="#334155">ошибка</text>
<text x="{left}" y="{height - 42}" font-size="12" font-family="Arial, sans-serif" fill="#475569">первая ошибка: {losses[0]:.6f}</text>
<text x="{left + 190}" y="{height - 42}" font-size="12" font-family="Arial, sans-serif" fill="#475569">последняя ошибка: {losses[-1]:.6f}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def heat_color(value, min_value, max_value):
    if max_value <= min_value:
        t = 0.0
    else:
        t = (value - min_value) / (max_value - min_value)
    t = max(0.0, min(1.0, t))

    # Синий -> голубой -> жёлтый -> красный.
    if t < 0.33:
        u = t / 0.33
        r, g, b = 20, int(65 + 140 * u), int(160 + 80 * u)
    elif t < 0.66:
        u = (t - 0.33) / 0.33
        r, g, b = int(20 + 230 * u), int(205 + 35 * u), int(240 - 180 * u)
    else:
        u = (t - 0.66) / 0.34
        r, g, b = 250, int(240 - 170 * u), int(60 - 40 * u)
    return f"rgb({r},{g},{b})"


def write_heatmap_svg(rows, path, max_rows=350):
    if not rows:
        raise ValueError("heatmap пустой")

    sampled = rows
    if len(rows) > max_rows:
        sampled = [rows[int(i * (len(rows) - 1) / (max_rows - 1))] for i in range(max_rows)]

    cols = len(sampled[0])
    values = [v for row in sampled for v in row]
    min_value = min(values)
    max_value = max(values)

    cell = 8
    label_h = 42
    width = max(720, cols * cell)
    height = label_h + len(sampled) * cell + 46

    rects = []
    for y, row in enumerate(sampled):
        for x, value in enumerate(row):
            rects.append(
                f'<rect x="{x * cell}" y="{label_h + y * cell}" width="{cell}" height="{cell}" '
                f'fill="{heat_color(value, min_value, max_value)}"/>'
            )

    subtitle = f"показано примеров: {len(sampled)}, скрытых нейронов: {cols}"
    if len(rows) != len(sampled):
        subtitle += f" (выборка из {len(rows)} строк)"

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="0" y="22" font-size="20" font-family="Arial, sans-serif" fill="#111827">Тепловая карта активаций скрытого слоя</text>
<text x="0" y="38" font-size="12" font-family="Arial, sans-serif" fill="#475569">{html.escape(subtitle)}</text>
{''.join(rects)}
<text x="0" y="{height - 18}" font-size="12" font-family="Arial, sans-serif" fill="#475569">минимальная активация: {min_value:.6f}</text>
<text x="250" y="{height - 18}" font-size="12" font-family="Arial, sans-serif" fill="#475569">максимальная активация: {max_value:.6f}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_index(output_dir):
    html_text = '''<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Визуализация нейросети MNIST</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #111827; background: #f8fafc; }
    h1 { margin-bottom: 8px; }
    section { margin-top: 24px; background: #fff; border: 1px solid #e5e7eb; padding: 16px; }
    img { max-width: 100%; height: auto; display: block; }
    code { background: #eef2ff; padding: 2px 5px; }
  </style>
</head>
<body>
  <h1>Визуализация нейросети MNIST</h1>
  <p>Страница построена по файлам <code>loss_history.txt</code> и <code>heatmap.txt</code>.</p>
  <section>
    <h2>График ошибки</h2>
    <img src="loss_curve.svg" alt="График ошибки">
  </section>
  <section>
    <h2>Активации скрытого слоя</h2>
    <img src="heatmap.svg" alt="Тепловая карта активаций скрытого слоя">
  </section>
</body>
</html>
'''
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--output-dir", default="visualization")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_path = input_dir / "loss_history.txt"
    heatmap_path = input_dir / "heatmap.txt"

    if not loss_path.exists():
        raise SystemExit(f"Не найден файл {loss_path}. Сначала запустите nn_classifier.")
    if not heatmap_path.exists():
        raise SystemExit(f"Не найден файл {heatmap_path}. Сначала запустите nn_classifier.")

    write_loss_svg(read_loss(loss_path), output_dir / "loss_curve.svg")
    write_heatmap_svg(read_heatmap(heatmap_path), output_dir / "heatmap.svg")
    write_index(output_dir)

    print(f"Визуализация сохранена в {output_dir}")


if __name__ == "__main__":
    main()
