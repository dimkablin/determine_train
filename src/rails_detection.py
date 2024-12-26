import cv2
import numpy as np


def detect_rails(
    img_bgr,
    canny1=80,
    canny2=200,
    hough_threshold=115,
    min_line_length=100,
    max_line_gap=40,
    max_slope_degrees=20
):
    """
    1) Ищет 'горизонтальные' или слабо наклонённые линии через Хафа.
    2) Сортирует их по вертикали (y).
    3) Парами соседних линий (0-1, 2-3, ...) считает пути, заливает их разными цветами.
    4) Возвращает:
       - итоговое изображение (numpy array, BGR),
       - маску путей (uint8), в которой фон=0, пути = 1,2,3...,
       - количество путей (int).
    """

    # ШАГ 1. Считываем изображение
    h, w = img_bgr.shape[:2]

    # Сохраним копию для рисования
    out_bgr = img_bgr.copy()

    # Перевод в градации серого + Canny
    cv2.blur(img_bgr, (7, 7))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny1, canny2)

    # ШАГ 2. Преобразование Хафа (версия Probabilistic)
    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    if lines_p is None:
        print("Не найдено линий HoughLinesP.")
        return None, None, 0

    # Фильтруем линии по углу (пример: допускаем наклон <= max_slope_degrees)
    acceptable_lines = []
    max_slope = np.tan(np.deg2rad(max_slope_degrees))
    for line in lines_p:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue  # вертикальная либо слишком крутая
        slope = abs(dy / dx)
        if slope <= max_slope:
            # Упорядочим, чтобы (x1, y1) был левее (x2, y2)
            if x2 < x1:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            acceptable_lines.append((x1, y1, x2, y2))

    if not acceptable_lines:
        print("Нет подходящих (полугоризонтальных) линий.")
        return None, None, 0

    # ШАГ 3. Группировка близких отрезков
    used = [False]*len(acceptable_lines)

    def avg_y(line):
        (xx1, yy1, xx2, yy2) = line
        return (yy1 + yy2)/2

    # Сортируем «сверху вниз»
    acceptable_lines.sort(key=avg_y)

    eps_y = 25  # "радиус" объединения по Y
    grouped_lines = []

    for i in range(len(acceptable_lines)):
        if used[i]:
            continue
        (x1_i, y1_i, x2_i, y2_i) = acceptable_lines[i]
        y_mid_i = (y1_i + y2_i) / 2

        group = [(x1_i, y1_i, x2_i, y2_i)]
        used[i] = True

        for j in range(i+1, len(acceptable_lines)):
            if used[j]:
                continue
            (x1_j, y1_j, x2_j, y2_j) = acceptable_lines[j]
            y_mid_j = (y1_j + y2_j) / 2
            if abs(y_mid_j - y_mid_i) < eps_y:
                group.append((x1_j, y1_j, x2_j, y2_j))
                used[j] = True

        # Сливаем группу в одну "среднюю" линию
        xs = []
        ys = []
        for (xa, ya, xb, yb) in group:
            xs.extend([xa, xb])
            ys.extend([ya, yb])
        x_min = min(xs)
        x_max = max(xs)
        y_mean = np.mean(ys)
        grouped_lines.append((int(x_min), int(y_mean), int(x_max), int(y_mean)))

    # Сортируем итоговые линии «сверху вниз»
    def line_y_mean(line):
        (xx1, yy1, xx2, yy2) = line
        return (yy1 + yy2)/2

    grouped_lines.sort(key=line_y_mean, reverse=True)

    # ШАГ 4. Заливаем участки (каждые 2 линии = 1 путь)
    overlay = out_bgr.copy()

    # Создаём маску путей (h, w), фон=0
    path_mask = np.zeros((h, w), dtype=np.uint8)

    colors = [
        (0,0,255),    # красный
        (0,255,0),    # зелёный
        (255,0,0),    # синий
        (0,255,255),  # жёлтый
        (255,0,255),  # фиолетовый
        (255,255,0),  # голубой
    ]

    alpha = 0.3

    num_paths = (len(grouped_lines) // 2)
    path_index = 0

    for i in range(0, len(grouped_lines) - 1, 2):
        path_index += 1
        (x1a, y1a, x2a, y2a) = grouped_lines[i]
        (x1b, y1b, x2b, y2b) = grouped_lines[i+1]

        ya = (y1a + y2a)//2
        yb = (y1b + y2b)//2
        y_top = min(ya, yb)
        y_bot = max(ya, yb)

        color = colors[i % len(colors)]

        pts = np.array([
            [0, y_top],
            [w, y_top],
            [w, y_bot],
            [0, y_bot]
        ], dtype=np.int32)

        # 1) Закрашиваем overlay цветом
        cv2.fillPoly(overlay, [pts], color)

        # 2) Заполняем path_mask индексом пути
        cv2.fillPoly(path_mask, [pts], path_index)

    # Смешиваем overlay и исходное (с прозрачностью alpha)
    cv2.addWeighted(overlay, alpha, out_bgr, 1 - alpha, 0, out_bgr)

    # ШАГ 5. Рисуем линии + подписи
    for idx, (x1, y1, x2, y2) in enumerate(grouped_lines):
        cv2.line(out_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label = f"Rail {idx+1}"
        cv2.putText(
            out_bgr,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # Возвращаем:
    # 1) Итоговое изображение (BGR) как np.array
    # 2) Маску путей (uint8)
    # 3) Количество путей
    return out_bgr, path_mask, num_paths
